package ollama

import (
	"context"
	"net/http"
	"reflect"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

const (
	// Thinking capability in /api/tags; /api/show has no thinking metadata.
	onOffModel = `{"model":"qwen3.5:4b","capabilities":["completion","thinking"],"details":{"context_length":4096}}`
	// Named thinking levels reported by /api/show.
	levelModel = `{"model":"gpt-oss:20b","capabilities":["completion","thinking"]}`
	plainModel = `{"model":"llama3.2:3b","capabilities":["completion"]}`
)

// generationServer serves the three models above; /api/show reports named
// levels for gpt-oss:20b only.
func generationServer(t *testing.T) *fakeServer {
	var s *fakeServer
	s = newFakeServer(t, map[string]http.HandlerFunc{
		"GET /api/tags": reply(200, tagsJSON(onOffModel, levelModel, plainModel)),
		"POST /api/show": func(w http.ResponseWriter, r *http.Request) {
			if s.lastBody()["model"] == "gpt-oss:20b" {
				reply(200, `{"capabilities":["completion","thinking"],"thinking":{"values":["low","medium","high"]}}`)(w, r)
				return
			}
			reply(200, `{"capabilities":["completion"]}`)(w, r)
		},
		"POST /api/chat": reply(200, okChat),
	})
	return s
}

// lastBody is the body of the request being handled (recorded first).
func (s *fakeServer) lastBody() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests[len(s.requests)-1].Body
}

func TestThinkParameterForReasoning(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		reasoning llm.Reasoning
		want      any // nil: no "think" field
	}{
		{"on/off default", "qwen3.5:4b", llm.Reasoning{}, false},
		{"on/off off", "qwen3.5:4b", llm.Reasoning{Effort: "off", Explicit: true}, false},
		{"on/off none", "qwen3.5:4b", llm.Reasoning{Effort: "none", Explicit: true}, false},
		{"on/off on", "qwen3.5:4b", llm.Reasoning{Effort: "on", Explicit: true}, true},
		{"on/off effort name", "qwen3.5:4b", llm.Reasoning{Effort: "high", Explicit: true}, true},
		{"levels default", "gpt-oss:20b", llm.Reasoning{}, false},
		{"levels named", "gpt-oss:20b", llm.Reasoning{Effort: "high", Explicit: true}, "high"},
		{"levels model default", "gpt-oss:20b", llm.Reasoning{Explicit: true}, nil},
		{"no thinking default", "llama3.2:3b", llm.Reasoning{}, nil},
		{"no thinking, explicit none", "llama3.2:3b", llm.Reasoning{Effort: "none", Explicit: true}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := generationServer(t)
			req := llm.Request{Model: tt.model, Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}}, Reasoning: tt.reasoning}
			if _, err := s.provider().Chat(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			chats := s.requestsTo("/api/chat")
			think, present := chats[0].Body["think"]
			if tt.want == nil && present || tt.want != nil && !reflect.DeepEqual(think, tt.want) {
				t.Errorf("think = %v (present %v), want %v", think, present, tt.want)
			}
		})
	}
}

func TestThinkingRejectedBeforeChat(t *testing.T) {
	tests := []struct {
		name, model, effort, want string
	}{
		{"no thinking capability", "llama3.2:3b", "high",
			"llama3.2:3b does not support thinking; turn reasoning off or choose another model."},
		{"unknown level", "gpt-oss:20b", "max",
			"gpt-oss:20b does not support thinking level 'max'. Supported: low, medium, high."},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := generationServer(t)
			req := llm.Request{Model: tt.model, Reasoning: llm.Reasoning{Effort: tt.effort, Explicit: true}}
			_, err := s.provider().Chat(context.Background(), req)
			if err == nil || err.Error() != tt.want {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
			if n := len(s.requestsTo("/api/chat")); n != 0 {
				t.Errorf("%d chat requests, want 0", n)
			}
		})
	}
}

func TestReasoningSupport(t *testing.T) {
	tests := []struct {
		model string
		want  llm.ReasoningSupport
	}{
		{"qwen3.5:4b", llm.ReasoningSupport{Values: []string{"off", "on"}, Known: true, Default: "off", NoModelDefault: true}},
		{"gpt-oss:20b", llm.ReasoningSupport{Values: []string{"low", "medium", "high"}, Known: true}},
		{"llama3.2:3b", llm.ReasoningSupport{Known: true}},
	}
	for _, tt := range tests {
		s := generationServer(t)
		p := s.provider()
		for range 2 {
			got, err := p.ReasoningSupport(context.Background(), tt.model)
			if err != nil || !reflect.DeepEqual(got, tt.want) {
				t.Errorf("%s: got %+v, %v; want %+v", tt.model, got, err, tt.want)
			}
		}
		if n := len(s.requestsTo("/api/show")); n != 1 {
			t.Errorf("%s: %d /api/show requests, want 1 (cached)", tt.model, n)
		}
	}

	t.Run("boolean thinking values", func(t *testing.T) {
		s := newFakeServer(t, map[string]http.HandlerFunc{
			"GET /api/tags":  reply(200, tagsJSON(oldOllama)),
			"POST /api/show": reply(200, `{"capabilities":["completion","thinking"],"thinking":{"values":[false,true]}}`),
		})
		got, err := s.provider().ReasoningSupport(context.Background(), "mistral:7b")
		want := llm.ReasoningSupport{Values: []string{"off", "on"}, Known: true, Default: "off", NoModelDefault: true}
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Errorf("got %+v, %v; want %+v", got, err, want)
		}
	})
}

func TestDefaultChatMakesNoExtraShowRequest(t *testing.T) {
	s := generationServer(t)
	if _, err := s.provider().Chat(context.Background(), llm.Request{Model: "qwen3.5:4b"}); err != nil {
		t.Fatal(err)
	}
	if n := len(s.requestsTo("/api/show")); n != 0 {
		t.Errorf("%d /api/show requests, want 0 (capabilities come from /api/tags)", n)
	}
}

func TestMaxOutputTokensAndInstructions(t *testing.T) {
	s := generationServer(t)
	p := s.provider()
	req := llm.Request{
		Model:           "llama3.2:3b",
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		Instructions:    llm.StyleInstructions(llm.StyleDetailed),
		MaxOutputTokens: 256,
	}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	body := s.requestsTo("/api/chat")[0].Body
	wantMessages := []any{
		map[string]any{"role": "system", "content": llm.StyleInstructions(llm.StyleDetailed)},
		map[string]any{"role": "user", "content": "Hi"},
	}
	if !reflect.DeepEqual(body["messages"], wantMessages) {
		t.Errorf("messages = %v, want %v", body["messages"], wantMessages)
	}
	if !reflect.DeepEqual(body["options"], map[string]any{"num_predict": float64(256)}) {
		t.Errorf("options = %v, want num_predict 256", body["options"])
	}

	limit, err := p.MaxOutputLimit(context.Background(), "qwen3.5:4b")
	if err != nil || limit != 4096 {
		t.Errorf("MaxOutputLimit = %d, %v; want 4096", limit, err)
	}
	if limit, _ := p.MaxOutputLimit(context.Background(), "llama3.2:3b"); limit != 0 {
		t.Errorf("unreported context length: limit = %d, want 0", limit)
	}
	if _, err := p.MaxOutputLimit(context.Background(), "absent:1b"); err == nil {
		t.Error("MaxOutputLimit for a missing model returned no error")
	}
}

func TestDoneReasonLength(t *testing.T) {
	for reason, want := range map[string]string{"length": "the max output token limit was reached", "stop": ""} {
		resp, err := parseChatResponse([]byte(`{"message":{"content":"partial"},"done":true,"done_reason":"` + reason + `"}`))
		if err != nil || resp.Text != "partial" || resp.DoneReason != reason || resp.IncompleteReason != want {
			t.Errorf("%s: resp=%+v err=%v", reason, resp, err)
		}
	}
}
