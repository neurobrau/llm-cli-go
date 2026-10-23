package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/llm/llmtest"
)

// sse formats one server-sent event with a compacted JSON payload.
func sse(eventType, data string) string {
	var buf bytes.Buffer
	if err := json.Compact(&buf, []byte(data)); err != nil {
		panic(err)
	}
	return "event: " + eventType + "\ndata: " + buf.String() + "\n\n"
}

func textDelta(text string) string {
	return sse("response.output_text.delta",
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"sequence_number":3,"delta":`+quote(text)+`}`)
}

func terminalEvent(eventType, response string) string {
	return sse(eventType, `{"type":"`+eventType+`","sequence_number":9,"response":`+response+`}`)
}

const (
	created        = `{"type":"response.created","sequence_number":0,"response":{"id":"resp_123","status":"in_progress","output":[],"error":null}}`
	reasoningEvent = `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"sequence_number":1,"delta":"SECRET SUMMARY"}`
	toolEvent      = `{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"sequence_number":2,"delta":"{\"x\":1}"}`
)

// streamReply writes the events one by one, flushing after each.
func streamReply(events ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			io.WriteString(w, e)
			w.(http.Flusher).Flush()
		}
	}
}

// streamProvider returns a provider for f whose responses are tracked.
func streamProvider(f *fakeAPI, cfg Config) (*Provider, *llmtest.Transport) {
	tr := &llmtest.Transport{Base: f.Client().Transport}
	cfg.HTTPClient = &http.Client{Transport: tr}
	cfg.APIKey = fakeKey
	cfg.BaseURL = f.URL + "/v1"
	return New(cfg), tr
}

func newStreamAPI(t *testing.T, handler http.HandlerFunc) *fakeAPI {
	return newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": handler})
}

func TestStreamRequestMatchesChat(t *testing.T) {
	tests := []struct {
		name string
		req  llm.Request
		want map[string]any
	}{
		{"built-in default", userRequest("gpt-6-luna", "Hi"),
			map[string]any{"reasoning": effort("none"), "max_output_tokens": nil, "instructions": nil}},
		{"model-default", llm.Request{Model: "gpt-6-luna", Messages: userRequest("", "Hi").Messages, Reasoning: llm.Reasoning{Explicit: true}},
			map[string]any{"reasoning": nil}},
		{"explicit, tokens, style", llm.Request{Model: "gpt-6-luna", Messages: userRequest("", "Hi").Messages,
			Reasoning: llm.Reasoning{Effort: "high", Explicit: true}, MaxOutputTokens: 300, Instructions: "Be brief."},
			map[string]any{"reasoning": effort("high"), "max_output_tokens": float64(300), "instructions": "Be brief."}},
		{"no reasoning model", userRequest("gpt-4o", "Hi"), map[string]any{"reasoning": nil}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newStreamAPI(t, streamReply(textDelta("ok"), terminalEvent("response.completed", completed)))
			p, _ := streamProvider(f, Config{})
			llmtest.Collect(t, p.Stream(context.Background(), tt.req))
			body := f.last().Body
			if body["stream"] != true || body["store"] != false || body["model"] != tt.req.Model {
				t.Errorf("body %v", body)
			}
			for k, v := range tt.want {
				if !reflect.DeepEqual(body[k], v) {
					t.Errorf("%s = %v, want %v", k, body[k], v)
				}
			}
			if !reflect.DeepEqual(body["input"], []any{map[string]any{"role": "user", "content": "Hi"}}) {
				t.Errorf("input %v", body["input"])
			}
		})
	}
}

func TestStreamCompleted(t *testing.T) {
	f := newStreamAPI(t, streamReply(
		sse("response.created", created), sse("response.reasoning_summary_text.delta", reasoningEvent),
		textDelta("Neural "), textDelta(""), sse("response.function_call_arguments.delta", toolEvent),
		textDelta("networks 🧠"), textDelta(" learn."),
		terminalEvent("response.completed", completed),
		textDelta("AFTER THE END"),
	))
	p, tr := streamProvider(f, Config{})
	r := llmtest.Collect(t, p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
	if !reflect.DeepEqual(r.Deltas, []string{"Neural ", "networks 🧠", " learn."}) || r.Reasoning != nil {
		t.Errorf("deltas %q reasoning %q", r.Deltas, r.Reasoning)
	}
	done, ok := r.Terminal.(llm.StreamCompleted)
	if !ok {
		t.Fatalf("terminal %#v", r.Terminal)
	}
	want := llm.Response{Text: "Neural networks 🧠 learn.", ID: "resp_123", Model: "gpt-6-luna-2026-09-01", ServiceTier: "default",
		Usage: &llm.Usage{InputTokens: intp(20), OutputTokens: intp(30), TotalTokens: intp(50),
			CachedInputTokens: intp(8), CacheWriteTokens: intp(2), ReasoningTokens: intp(12)}}
	if !reflect.DeepEqual(done.Response, want) {
		t.Errorf("response %+v, want %+v", done.Response, want)
	}
	if tr.Open() != 0 || f.count() != 1 {
		t.Errorf("%d open bodies, %d requests", tr.Open(), f.count())
	}
}

func TestStreamCompletedWithoutDeltasUsesOutputText(t *testing.T) {
	f := newStreamAPI(t, streamReply(terminalEvent("response.completed", completed)))
	p, _ := streamProvider(f, Config{})
	r := llmtest.Collect(t, p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
	if done, ok := r.Terminal.(llm.StreamCompleted); !ok || done.Response.Text != "Neural networks learn from data." {
		t.Errorf("terminal %#v", r.Terminal)
	}
}

func TestStreamTerminalOutcomes(t *testing.T) {
	incomplete := func(reason string) string {
		return `{"id":"r","status":"incomplete","service_tier":"flex","incomplete_details":{"reason":"` + reason + `"},` +
			`"output":[],"usage":{"input_tokens":5,"output_tokens":9,"total_tokens":14}}`
	}
	tests := []struct {
		name   string
		events []string
		want   llm.StreamEvent
	}{
		{"token limit", []string{textDelta("Part"), terminalEvent("response.incomplete", incomplete("max_output_tokens"))},
			llm.StreamIncomplete{Reason: "the max output token limit was reached", PartialText: "Part", ServiceTier: "flex",
				Usage: &llm.Usage{InputTokens: intp(5), OutputTokens: intp(9), TotalTokens: intp(14)}}},
		{"content filter", []string{terminalEvent("response.incomplete", incomplete("content_filter"))},
			llm.StreamIncomplete{Reason: "the content filter stopped the response", ServiceTier: "flex",
				Usage: &llm.Usage{InputTokens: intp(5), OutputTokens: intp(9), TotalTokens: intp(14)}}},
		{"failed", []string{textDelta("Part"), terminalEvent("response.failed",
			`{"id":"r","status":"failed","error":{"code":"server_error","message":"The model `+fakeKey+` crashed."}}`)},
			llm.StreamFailed{Err: apiError("OpenAI response failed: The model [redacted] crashed."), PartialText: "Part"}},
		{"error event", []string{textDelta("Part"), sse("error", `{"type":"error","code":"overloaded","message":"Too busy","sequence_number":4}`)},
			llm.StreamFailed{Err: apiError("OpenAI stream error: Too busy"), PartialText: "Part"}},
		{"error payload", []string{textDelta("Part"), "data: {\"error\":{\"message\":\"Server exploded\"}}\n\n"},
			llm.StreamFailed{Err: apiError("OpenAI stream error: Server exploded"), PartialText: "Part"}},
		{"end without terminal event", []string{textDelta("Part")},
			llm.StreamFailed{Err: apiError("OpenAI stream ended before the response was complete."), PartialText: "Part"}},
		{"non-string delta", []string{textDelta("Part"), sse("response.output_text.delta", `{"type":"response.output_text.delta","delta":5}`)},
			llm.StreamFailed{Err: apiError("Invalid stream event from OpenAI: text delta is not a string."), PartialText: "Part"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newStreamAPI(t, streamReply(tt.events...))
			p, tr := streamProvider(f, Config{})
			r := llmtest.Collect(t, p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
			if !reflect.DeepEqual(r.Terminal, tt.want) {
				t.Errorf("terminal %#v\nwant     %#v", r.Terminal, tt.want)
			}
			if tr.Open() != 0 || f.count() != 1 {
				t.Errorf("%d open bodies, %d requests", tr.Open(), f.count())
			}
		})
	}
}

func TestStreamHTTPErrors(t *testing.T) {
	tests := []struct {
		status int
		kind   llm.ErrorKind
	}{
		{401, llm.KindAuthentication}, {403, llm.KindAuthentication}, {429, llm.KindRateLimit},
		{400, llm.KindAPI}, {500, llm.KindAPI},
	}
	for _, tt := range tests {
		f := newStreamAPI(t, reply(tt.status, `{"error":{"message":"bad `+fakeKey+`","type":"x"}}`))
		p, tr := streamProvider(f, Config{ChatMaxRetries: new(int)})
		r := llmtest.Collect(t, p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
		failed, ok := r.Terminal.(llm.StreamFailed)
		if !ok || kindOf(t, failed.Err) != tt.kind || strings.Contains(failed.Err.Error(), fakeKey) {
			t.Errorf("HTTP %d: %#v", tt.status, r.Terminal)
		}
		if tr.Open() != 0 {
			t.Errorf("HTTP %d: open body", tt.status)
		}
	}
}

func TestStreamRetriesBeforeTheStreamOnly(t *testing.T) {
	calls := 0
	f := newStreamAPI(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			reply(503, `{"error":{"message":"busy"}}`)(w, r)
			return
		}
		streamReply(textDelta("Part"))(w, r) // then the connection ends
	})
	p, _ := streamProvider(f, Config{})
	r := llmtest.Collect(t, p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
	if _, ok := r.Terminal.(llm.StreamFailed); !ok || f.count() != 2 || r.Text() != "Part" {
		t.Errorf("%d requests, terminal %#v", f.count(), r.Terminal)
	}
}

func TestStreamRejectedReasoning(t *testing.T) {
	t.Run("built-in default falls back once", func(t *testing.T) {
		var f *fakeAPI
		f = newStreamAPI(t, func(w http.ResponseWriter, r *http.Request) {
			if !rejectReasoning(w, r, f.last().Body) {
				streamReply(textDelta("ok"), terminalEvent("response.completed", completed))(w, r)
			}
		})
		p, tr := streamProvider(f, Config{})
		for range 2 {
			r := llmtest.Collect(t, p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
			if _, ok := r.Terminal.(llm.StreamCompleted); !ok || r.Text() != "ok" {
				t.Errorf("terminal %#v", r.Terminal)
			}
		}
		bodies := f.bodies()
		if len(bodies) != 3 || !reflect.DeepEqual(bodies[0]["reasoning"], effort("none")) || bodies[1]["reasoning"] != nil || bodies[2]["reasoning"] != nil {
			t.Errorf("requests %v", bodies)
		}
		if tr.Open() != 0 {
			t.Error("open body")
		}
	})
	t.Run("explicit effort is reported", func(t *testing.T) {
		var f *fakeAPI
		f = newStreamAPI(t, func(w http.ResponseWriter, r *http.Request) { rejectReasoning(w, r, f.last().Body) })
		p, _ := streamProvider(f, Config{})
		req := userRequest("gpt-6-luna", "Hi")
		req.Reasoning = llm.Reasoning{Effort: "high", Explicit: true}
		r := llmtest.Collect(t, p.Stream(context.Background(), req))
		failed, ok := r.Terminal.(llm.StreamFailed)
		if !ok || f.count() != 1 || !strings.HasPrefix(failed.Err.Error(), "gpt-6-luna does not accept reasoning effort 'high' (") {
			t.Errorf("%d requests, terminal %#v", f.count(), r.Terminal)
		}
	})
	t.Run("other 400 errors are not rejections", func(t *testing.T) {
		f := newStreamAPI(t, reply(400, `{"error":{"message":"bad input","param":"input"}}`))
		p, _ := streamProvider(f, Config{})
		llmtest.Collect(t, p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
		if f.count() != 1 {
			t.Errorf("%d requests", f.count())
		}
	})
}

func TestStreamMissingKeyAndConnection(t *testing.T) {
	r := llmtest.Collect(t, New(Config{}).Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
	if failed, ok := r.Terminal.(llm.StreamFailed); !ok || !errors.Is(failed.Err, ErrMissingAPIKey) {
		t.Errorf("missing key: %#v", r.Terminal)
	}
	p := New(Config{APIKey: fakeKey, BaseURL: "http://127.0.0.1:1/v1", ChatMaxRetries: new(int)})
	r = llmtest.Collect(t, p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
	if failed, ok := r.Terminal.(llm.StreamFailed); !ok || kindOf(t, failed.Err) != llm.KindConnection {
		t.Errorf("connection: %#v", r.Terminal)
	}
}

// stallAfter sends events, then holds the connection open until the
// client closes it; closed is signaled then.
func stallAfter(closed chan<- struct{}, events ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		streamReply(events...)(w, r)
		<-r.Context().Done()
		closed <- struct{}{}
	}
}

func TestStreamIdleTimeout(t *testing.T) {
	closed := make(chan struct{}, 1)
	f := newStreamAPI(t, stallAfter(closed, textDelta("Part")))
	p, tr := streamProvider(f, Config{ChatTimeout: 50 * time.Millisecond})
	r := llmtest.Collect(t, p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")))
	failed, ok := r.Terminal.(llm.StreamFailed)
	if !ok || kindOf(t, failed.Err) != llm.KindTimeout || failed.PartialText != "Part" || f.count() != 1 {
		t.Errorf("%d requests, terminal %#v", f.count(), r.Terminal)
	}
	<-closed
	if tr.Open() != 0 {
		t.Error("open body")
	}
}

func TestStreamIdleTimeoutIgnoresSlowConsumers(t *testing.T) {
	f := newStreamAPI(t, streamReply(textDelta("a"), textDelta("b"), terminalEvent("response.completed", completed)))
	p, _ := streamProvider(f, Config{ChatTimeout: 30 * time.Millisecond})
	var got []string
	for event := range p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")) {
		if d, ok := event.(llm.TextDelta); ok {
			got = append(got, d.Text)
			time.Sleep(60 * time.Millisecond) // longer than the idle timeout
			continue
		}
		if _, ok := event.(llm.StreamCompleted); !ok {
			t.Errorf("terminal %#v", event)
		}
	}
	if strings.Join(got, "") != "ab" {
		t.Errorf("deltas %q", got)
	}
}

func TestStreamCancellation(t *testing.T) {
	closed := make(chan struct{}, 1)
	f := newStreamAPI(t, stallAfter(closed, textDelta("Part")))
	p, tr := streamProvider(f, Config{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var events []llm.StreamEvent
	for event := range p.Stream(ctx, userRequest("gpt-6-luna", "Hi")) {
		events = append(events, event)
		if _, ok := event.(llm.TextDelta); ok {
			cancel() // Ctrl+C while the server stalls
		}
	}
	if len(events) != 2 {
		t.Fatalf("events %#v", events)
	}
	failed, ok := events[1].(llm.StreamFailed)
	if !ok || !errors.Is(failed.Err, context.Canceled) || failed.PartialText != "Part" {
		t.Errorf("terminal %#v", events[1])
	}
	<-closed
	if tr.Open() != 0 || f.count() != 1 {
		t.Errorf("%d open bodies, %d requests", tr.Open(), f.count())
	}
}

func TestStreamConsumerStopsEarly(t *testing.T) {
	closed := make(chan struct{}, 1)
	f := newStreamAPI(t, stallAfter(closed, textDelta("one"), textDelta("two")))
	p, tr := streamProvider(f, Config{})
	for event := range p.Stream(context.Background(), userRequest("gpt-6-luna", "Hi")) {
		if _, ok := event.(llm.TextDelta); !ok {
			t.Errorf("event %#v", event)
		}
		break
	}
	<-closed // the server saw the connection close
	if tr.Open() != 0 {
		t.Error("open body")
	}
}
