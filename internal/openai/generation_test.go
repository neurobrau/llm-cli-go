package openai

import (
	"context"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

func effort(e string) any { return map[string]any{"effort": e} }

func (f *fakeAPI) bodies() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []map[string]any
	for _, r := range f.requests {
		out = append(out, r.Body)
	}
	return out
}

func TestReasoningInRequest(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		reasoning llm.Reasoning
		want      any // nil: no "reasoning" field
	}{
		// Regression: NeuroBräu's default for gpt-6-luna is "none", not the
		// model's own default ("medium").
		{"gpt-6-luna default", "gpt-6-luna", llm.Reasoning{}, effort("none")},
		{"dated snapshot default", "gpt-6-luna-2026-09-01", llm.Reasoning{}, effort("none")},
		{"default without none", "gpt-5", llm.Reasoning{}, nil},
		{"non-reasoning model", "gpt-4o", llm.Reasoning{}, nil},
		{"unknown model default", "my-new-model", llm.Reasoning{}, nil},
		{"model default", "gpt-6-luna", llm.Reasoning{Explicit: true}, nil},
		{"explicit high", "gpt-6-luna", llm.Reasoning{Effort: "high", Explicit: true}, effort("high")},
		{"explicit none", "gpt-6-luna", llm.Reasoning{Effort: "none", Explicit: true}, effort("none")},
		{"unknown model, explicit effort", "my-new-model", llm.Reasoning{Effort: "low", Explicit: true}, effort("low")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": reply(200, completed)})
			req := userRequest(tt.model, "Hi")
			req.Reasoning = tt.reasoning
			if _, err := f.provider(Config{}).Chat(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			body := f.last().Body
			if got := body["reasoning"]; !reflect.DeepEqual(got, tt.want) {
				t.Errorf("reasoning = %v, want %v", got, tt.want)
			}
			if body["store"] != false {
				t.Errorf("store = %v, want false", body["store"])
			}
		})
	}
}

const reasoningRejected = `{"error":{"message":"Unsupported parameter: 'reasoning.effort' is not supported with this model.","type":"invalid_request_error","param":"reasoning.effort","code":"unsupported_parameter"}}`

// rejectReasoning answers 400 to any request with a reasoning field.
func rejectReasoning(w http.ResponseWriter, r *http.Request, body map[string]any) bool {
	if _, ok := body["reasoning"]; ok {
		reply(400, reasoningRejected)(w, r)
		return true
	}
	return false
}

func newRejectingAPI(t *testing.T) *fakeAPI {
	var f *fakeAPI
	f = newFakeAPI(t, map[string]http.HandlerFunc{
		"POST /v1/responses": func(w http.ResponseWriter, r *http.Request) {
			if !rejectReasoning(w, r, f.last().Body) {
				reply(200, completed)(w, r)
			}
		},
	})
	return f
}

func TestRejectedDefaultReasoningFallsBackOnce(t *testing.T) {
	f := newRejectingAPI(t)
	p := f.provider(Config{})
	for i := range 2 {
		resp, err := p.Chat(context.Background(), userRequest("gpt-6-luna", "Hi"))
		if err != nil || resp.ID != "resp_123" {
			t.Fatalf("chat %d: resp=%+v err=%v", i, resp, err)
		}
	}
	bodies := f.bodies()
	if len(bodies) != 3 {
		t.Fatalf("%d requests, want 3 (rejected default, fallback, then no reasoning)", len(bodies))
	}
	wantReasoning := []any{effort("none"), nil, nil}
	for i, body := range bodies {
		if !reflect.DeepEqual(body["reasoning"], wantReasoning[i]) {
			t.Errorf("request %d: reasoning = %v, want %v", i, body["reasoning"], wantReasoning[i])
		}
		if body["store"] != false {
			t.Errorf("request %d: store = %v, want false", i, body["store"])
		}
	}
	support, _ := p.ReasoningSupport(context.Background(), "gpt-6-luna")
	if !support.Known || len(support.Values) != 0 {
		t.Errorf("support after rejection = %+v, want known with no values", support)
	}
}

func TestRejectedExplicitReasoningIsReported(t *testing.T) {
	f := newRejectingAPI(t)
	req := userRequest("gpt-6-luna", "Hi")
	req.Reasoning = llm.Reasoning{Effort: "high", Explicit: true}
	_, err := f.provider(Config{}).Chat(context.Background(), req)
	want := "gpt-6-luna does not accept reasoning effort 'high' (OpenAI API error (HTTP 400): " +
		"Unsupported parameter: 'reasoning.effort' is not supported with this model.). " +
		"Choose another value with --reasoning-effort."
	if err == nil || kindOf(t, err) != llm.KindAPI || err.Error() != want {
		t.Fatalf("err = %v, want %q", err, want)
	}
	if n := f.count(); n != 1 {
		t.Errorf("%d requests, want 1 (no fallback for an explicit choice)", n)
	}
}

func TestOtherErrorsDoNotTriggerReasoningFallback(t *testing.T) {
	tests := map[string]http.HandlerFunc{
		"400 for another parameter": reply(400, `{"error":{"message":"Invalid 'input'.","param":"input"}}`),
		"400 without parameter":     reply(400, `{"error":{"message":"Bad request."}}`),
		"401":                       reply(401, `{"error":{"message":"bad key","param":"reasoning"}}`),
		"429":                       reply(429, `{"error":{"message":"quota","param":"reasoning"}}`),
	}
	for name, handler := range tests {
		t.Run(name, func(t *testing.T) {
			zero := 0
			f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": handler})
			_, err := f.provider(Config{ChatMaxRetries: &zero}).Chat(context.Background(), userRequest("gpt-6-luna", "Hi"))
			if err == nil {
				t.Fatal("no error")
			}
			if n := f.count(); n != 1 {
				t.Errorf("%d requests, want 1", n)
			}
		})
	}
}

func TestMaxOutputTokensAndInstructions(t *testing.T) {
	f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": reply(200, completed)})
	p := f.provider(Config{})

	plain := userRequest("my-model", "Hi")
	if _, err := p.Chat(context.Background(), plain); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"max_output_tokens", "instructions", "reasoning"} {
		if _, ok := f.last().Body[key]; ok {
			t.Errorf("plain request has %q", key)
		}
	}

	styled := userRequest("my-model", "Hi")
	styled.MaxOutputTokens = 256
	styled.Instructions = llm.StyleInstructions(llm.StyleBrief)
	if _, err := p.Chat(context.Background(), styled); err != nil {
		t.Fatal(err)
	}
	body := f.last().Body
	if body["max_output_tokens"] != float64(256) {
		t.Errorf("max_output_tokens = %v, want 256", body["max_output_tokens"])
	}
	if body["instructions"] != llm.StyleInstructions(llm.StyleBrief) {
		t.Errorf("instructions = %v", body["instructions"])
	}
	if input := body["input"].([]any); len(input) != 1 {
		t.Errorf("instructions leaked into the input messages: %v", input)
	}
}

func TestReasoningSupportAndLimits(t *testing.T) {
	p := New(Config{}) // no key: the table needs no request
	tests := []struct {
		model        string
		wantValues   []string
		wantKnown    bool
		wantDefault  string
		wantModelDef string
		wantLimit    int
	}{
		{"gpt-6-luna", []string{"none", "low", "medium", "high", "xhigh", "max"}, true, "none", "medium", 0},
		{"gpt-6-luna-2026-09-01", []string{"none", "low", "medium", "high", "xhigh", "max"}, true, "none", "medium", 0},
		{"gpt-5.2", []string{"none", "low", "medium", "high", "xhigh"}, true, "none", "", 128_000},
		{"gpt-5-2025-08-07", []string{"minimal", "low", "medium", "high"}, true, "", "", 128_000},
		{"o4-mini", []string{"low", "medium", "high"}, true, "", "", 100_000},
		{"gpt-4o", nil, true, "", "", 16_384},
		{"gpt-4.1-nano", nil, true, "", "", 32_768},
		{"some-future-model", nil, false, "", "", 0},
	}
	for _, tt := range tests {
		support, err := p.ReasoningSupport(context.Background(), tt.model)
		limit, _ := p.MaxOutputLimit(context.Background(), tt.model)
		if err != nil || !slices.Equal(support.Values, tt.wantValues) || support.Known != tt.wantKnown || support.Default != tt.wantDefault || support.ModelDefault != tt.wantModelDef || support.NoModelDefault {
			t.Errorf("%s: support = %+v, %v", tt.model, support, err)
		}
		if limit != tt.wantLimit {
			t.Errorf("%s: limit = %d, want %d", tt.model, limit, tt.wantLimit)
		}
	}
}

func TestErrorTextNeverShowsTheKey(t *testing.T) {
	echo := `{"error":{"message":"Bad request for key ` + fakeKey + ` and token sk-proj-ABCDEFGH12345678.","param":"input"}}`
	tests := map[string]http.HandlerFunc{
		"status error":    reply(400, echo),
		"failed response": reply(200, `{"status":"failed","error":{"message":"key `+fakeKey+` sk-proj-ABCDEFGH12345678"},"output":[]}`),
	}
	for name, handler := range tests {
		t.Run(name, func(t *testing.T) {
			f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": handler})
			_, err := f.provider(Config{}).Chat(context.Background(), userRequest("my-model", "Hi"))
			if err == nil {
				t.Fatal("no error")
			}
			if strings.Contains(err.Error(), fakeKey) || strings.Contains(err.Error(), "ABCDEFGH12345678") {
				t.Errorf("message exposes a key: %q", err)
			}
			if !strings.Contains(err.Error(), "[redacted]") {
				t.Errorf("message = %q, want the key replaced", err)
			}
		})
	}
}
