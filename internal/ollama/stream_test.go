package ollama

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/llm/llmtest"
)

// ndjson writes the chunks one by one, flushing after each.
func ndjson(chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, c := range chunks {
			io.WriteString(w, c)
			w.(http.Flusher).Flush()
		}
	}
}

func newStreamServer(t *testing.T, chat http.HandlerFunc) *fakeServer {
	return newFakeServer(t, map[string]http.HandlerFunc{
		"GET /api/tags":  reply(200, tagsJSON(localQwen, localLlama, cloudTag)),
		"POST /api/chat": chat,
	})
}

func streamProvider(s *fakeServer, cfg Config) (*Provider, *llmtest.Transport) {
	tr := &llmtest.Transport{Base: s.Client().Transport}
	cfg.BaseURL = s.URL
	cfg.HTTPClient = &http.Client{Transport: tr}
	return New(cfg), tr
}

func chatReq(model string) llm.Request {
	return llm.Request{Model: model, Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}}}
}

const (
	finalLine = `{"model":"qwen3.5:4b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",` +
		`"prompt_eval_count":12,"eval_count":5,"total_duration":2500000000,"load_duration":1000000000,` +
		`"prompt_eval_duration":300000000,"eval_duration":1200000000}` + "\n"
	lengthLine = `{"model":"qwen3.5:4b","message":{"role":"assistant","content":""},"done":true,"done_reason":"length","prompt_eval_count":3,"eval_count":7}` + "\n"
)

func content(text string) string {
	return `{"model":"qwen3.5:4b","message":{"role":"assistant","content":` + jsonString(text) + `},"done":false}` + "\n"
}

func thinking(text string) string {
	return `{"model":"qwen3.5:4b","message":{"role":"assistant","content":"","thinking":` + jsonString(text) + `},"done":false}` + "\n"
}

func jsonString(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`, "\r", `\r`)
	return `"` + r.Replace(s) + `"`
}

func TestStreamRequestAndCompletion(t *testing.T) {
	s := newStreamServer(t, ndjson(thinking("Let me think."), content("Hel"), content(""), content("lo, "), content("Wörld 🌍!"), finalLine))
	p, tr := streamProvider(s, Config{})
	req := chatReq("qwen3.5:4b")
	req.Instructions = "Be brief."
	req.MaxOutputTokens = 64
	r := llmtest.Collect(t, p.Stream(context.Background(), req))

	if !reflect.DeepEqual(r.Deltas, []string{"Hel", "lo, ", "Wörld 🌍!"}) || !reflect.DeepEqual(r.Reasoning, []string{"Let me think."}) {
		t.Errorf("deltas %q reasoning %q", r.Deltas, r.Reasoning)
	}
	want := llm.Response{Text: "Hello, Wörld 🌍!", Reasoning: "Let me think.", Model: "qwen3.5:4b", DoneReason: "stop",
		Usage:  &llm.Usage{InputTokens: intPtr(12), OutputTokens: intPtr(5), TotalTokens: intPtr(17)},
		Timing: llm.Timing{Total: 2500 * time.Millisecond, Load: time.Second, PromptEval: 300 * time.Millisecond, Eval: 1200 * time.Millisecond}}
	if done, ok := r.Terminal.(llm.StreamCompleted); !ok || !reflect.DeepEqual(done.Response, want) {
		t.Errorf("terminal %#v", r.Terminal)
	}
	body := s.requestsTo("/api/chat")[0].Body
	wantBody := map[string]any{"model": "qwen3.5:4b", "stream": true, "think": false,
		"options":  map[string]any{"num_predict": float64(64)},
		"messages": []any{map[string]any{"role": "system", "content": "Be brief."}, map[string]any{"role": "user", "content": "Hi"}}}
	if !reflect.DeepEqual(body, wantBody) {
		t.Errorf("body %v", body)
	}
	if tr.Open() != 0 {
		t.Error("open body")
	}
}

// chunkReader returns the chunks one per Read.
type chunkReader struct{ chunks []string }

func (c *chunkReader) Read(p []byte) (int, error) {
	if len(c.chunks) == 0 {
		return 0, io.EOF
	}
	n := copy(p, c.chunks[0])
	c.chunks[0] = c.chunks[0][n:]
	if c.chunks[0] == "" {
		c.chunks = c.chunks[1:]
	}
	return n, nil
}

func readAll(t *testing.T, chunks ...string) llmtest.Result {
	t.Helper()
	events := func(yield func(llm.StreamEvent) bool) {
		s := chatStream{model: "m", yield: yield}
		idle := llm.NewIdleTimer(time.Hour, func() {})
		defer idle.Stop()
		if terminal, ok := s.read(&chunkReader{chunks: chunks}, idle); ok {
			yield(terminal)
		}
	}
	return llmtest.Collect(t, events)
}

// TestNDJSONSplitAtEveryBoundary splits a body with multi-byte characters,
// CRLF line ends, blank lines and several objects per line at every byte
// offset, and into single bytes.
func TestNDJSONSplitAtEveryBoundary(t *testing.T) {
	body := thinking("ü") + "\r\n" + strings.TrimSuffix(content("Grüße, "), "\n") + "\r\n" + content("世界 🎉") + "\n  \n" +
		content("\\n\"quoted\"") + strings.TrimSuffix(finalLine, "\n") // no newline after the last line
	wantText := `Grüße, 世界 🎉\n"quoted"`
	check := func(name string, r llmtest.Result) {
		if r.Text() != wantText || strings.Join(r.Reasoning, "") != "ü" {
			t.Errorf("%s: text %q reasoning %q", name, r.Text(), r.Reasoning)
		}
		if _, ok := r.Terminal.(llm.StreamCompleted); !ok {
			t.Errorf("%s: terminal %#v", name, r.Terminal)
		}
	}
	for i := 0; i <= len(body); i++ {
		check(fmt.Sprintf("split at %d", i), readAll(t, body[:i], body[i:]))
	}
	single := make([]string, len(body))
	for i := 0; i < len(body); i++ { // bytes, not runes
		single[i] = body[i : i+1]
	}
	check("single bytes", readAll(t, single...))
}

func TestThinkTagsAcrossChunks(t *testing.T) {
	tests := []struct {
		name      string
		pieces    []string
		text      string
		reasoning string
	}{
		{"split tags", []string{"<", "thi", "nk>secret", " plan</th", "ink>\n\nAnswer", " more"}, "Answer more", "secret plan"},
		{"leading space", []string{"  <think>", "x</think>", "  ", "\nOK"}, "OK", "x"},
		{"no block", []string{"<", "thx>", " fine"}, "<thx> fine", ""},
		{"tag later in the answer", []string{"Hi <think>x</think>"}, "Hi <think>x</think>", ""},
		{"partial tag at the end", []string{"<th"}, "<th", ""},
		{"unclosed block", []string{"<think>never ", "closed</thi"}, "", "never clos"},
		{"multi-byte reasoning", []string{"<think>äöü€", "🎉🎉</think>Ja"}, "Ja", "äöü€🎉🎉"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var chunks []string
			for _, p := range tt.pieces {
				chunks = append(chunks, content(p))
			}
			r := readAll(t, append(chunks, finalLine)...)
			if r.Text() != tt.text || strings.Join(r.Reasoning, "") != tt.reasoning {
				t.Errorf("text %q reasoning %q; want %q %q", r.Text(), strings.Join(r.Reasoning, ""), tt.text, tt.reasoning)
			}
			if done, ok := r.Terminal.(llm.StreamCompleted); !ok || done.Response.Text != tt.text {
				t.Errorf("terminal %#v", r.Terminal)
			}
		})
	}
}

func TestThinkFilterBoundsMemory(t *testing.T) {
	var f thinkFilter
	f.feed("<think>")
	long := strings.Repeat("reasoning ", 1000)
	total := 0
	for range 1000 {
		answer, reasoning := f.feed(long)
		if answer != "" {
			t.Fatal("reasoning leaked into the answer")
		}
		total += len(reasoning)
		if len(f.pending) >= len(thinkClose) {
			t.Fatalf("holding back %d bytes", len(f.pending))
		}
	}
	if total+len(f.pending) != 1000*len(long) {
		t.Errorf("reasoning bytes %d", total)
	}
}

func TestStreamTerminalOutcomes(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		want   llm.StreamEvent
	}{
		{"token limit", []string{content("Part"), lengthLine}, llm.StreamIncomplete{Reason: "the max output token limit was reached",
			PartialText: "Part", Usage: &llm.Usage{InputTokens: intPtr(3), OutputTokens: intPtr(7), TotalTokens: intPtr(10)}}},
		{"error line", []string{content("Part"), `{"error":"model crashed"}` + "\n"},
			llm.StreamFailed{Err: &llm.Error{Kind: llm.KindAPI, Message: "Ollama error: model crashed"}, PartialText: "Part"}},
		{"not JSON", []string{content("Part"), "{oops\n"},
			llm.StreamFailed{Err: invalidResponse("stream line is not valid JSON"), PartialText: "Part"}},
		{"invalid UTF-8", []string{"{\"message\":{\"content\":\"\xff\"}}\n"},
			llm.StreamFailed{Err: invalidResponse("stream line is not valid JSON")}},
		{"not an object", []string{"[1]\n"}, llm.StreamFailed{Err: invalidResponse("stream line is not a JSON object")}},
		{"content not a string", []string{content("Part"), `{"message":{"content":5},"done":false}` + "\n"},
			llm.StreamFailed{Err: invalidResponse("stream line has no valid 'message.content'"), PartialText: "Part"}},
		{"content null", []string{`{"message":{"content":null}}` + "\n"},
			llm.StreamFailed{Err: invalidResponse("stream line has no valid 'message.content'")}},
		{"message not an object", []string{`{"message":"hi"}` + "\n"},
			llm.StreamFailed{Err: invalidResponse("stream line has no valid 'message.content'")}},
		{"end without done", []string{content("Part")}, llm.StreamFailed{Err: &llm.Error{Kind: llm.KindConnection,
			Message: "Ollama stream ended before the response was complete."}, PartialText: "Part"}},
		{"done is not true", []string{content("Part"), `{"done":"true"}` + "\n"}, llm.StreamFailed{Err: &llm.Error{Kind: llm.KindConnection,
			Message: "Ollama stream ended before the response was complete."}, PartialText: "Part"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newStreamServer(t, ndjson(tt.chunks...))
			p, tr := streamProvider(s, Config{})
			r := llmtest.Collect(t, p.Stream(context.Background(), chatReq("llama3.2:3b")))
			if !reflect.DeepEqual(r.Terminal, tt.want) {
				t.Errorf("terminal %#v\nwant     %#v", r.Terminal, tt.want)
			}
			if tr.Open() != 0 {
				t.Error("open body")
			}
		})
	}
}

func TestStreamRequestErrors(t *testing.T) {
	t.Run("cloud model is refused before any chat request", func(t *testing.T) {
		s := newStreamServer(t, ndjson(finalLine))
		p, _ := streamProvider(s, Config{})
		r := llmtest.Collect(t, p.Stream(context.Background(), chatReq("glm-4.7:cloud")))
		if f, ok := r.Terminal.(llm.StreamFailed); !ok || !errors.Is(f.Err, ErrCloudModel) || len(s.requestsTo("/api/chat")) != 0 {
			t.Errorf("terminal %#v", r.Terminal)
		}
	})
	t.Run("404 gives the install hint", func(t *testing.T) {
		s := newStreamServer(t, reply(404, `{"error":"model 'llama3.2:3b' not found"}`))
		p, tr := streamProvider(s, Config{})
		r := llmtest.Collect(t, p.Stream(context.Background(), chatReq("llama3.2:3b")))
		f, ok := r.Terminal.(llm.StreamFailed)
		if !ok || !strings.HasSuffix(f.Err.Error(), "Install it yourself with: ollama pull llama3.2:3b") || tr.Open() != 0 {
			t.Errorf("terminal %#v", r.Terminal)
		}
	})
	t.Run("HTTP 500", func(t *testing.T) {
		s := newStreamServer(t, reply(500, `{"error":"boom"}`))
		p, _ := streamProvider(s, Config{})
		r := llmtest.Collect(t, p.Stream(context.Background(), chatReq("llama3.2:3b")))
		if f, ok := r.Terminal.(llm.StreamFailed); !ok || f.Err.Error() != "Ollama API error (HTTP 500): boom" || kindOf(t, f.Err) != llm.KindAPI {
			t.Errorf("terminal %#v", r.Terminal)
		}
	})
	t.Run("connection refused", func(t *testing.T) {
		p := New(Config{BaseURL: "http://127.0.0.1:1"})
		r := llmtest.Collect(t, p.Stream(context.Background(), chatReq("llama3.2:3b")))
		if f, ok := r.Terminal.(llm.StreamFailed); !ok || kindOf(t, f.Err) != llm.KindConnection {
			t.Errorf("terminal %#v", r.Terminal)
		}
	})
	t.Run("connection drops mid-stream", func(t *testing.T) {
		s := newStreamServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Length", "10000") // promises more than it sends
			io.WriteString(w, content("Part"))
			w.(http.Flusher).Flush()
			conn, _, _ := w.(http.Hijacker).Hijack()
			conn.Close()
		})
		p, tr := streamProvider(s, Config{})
		r := llmtest.Collect(t, p.Stream(context.Background(), chatReq("llama3.2:3b")))
		if f, ok := r.Terminal.(llm.StreamFailed); !ok || kindOf(t, f.Err) != llm.KindConnection || f.PartialText != "Part" || tr.Open() != 0 {
			t.Errorf("terminal %#v", r.Terminal)
		}
	})
}

// stallAfter sends chunks, then holds the connection until the client
// closes it, which is signaled on closed.
func stallAfter(closed chan<- struct{}, chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ndjson(chunks...)(w, r)
		<-r.Context().Done()
		closed <- struct{}{}
	}
}

func TestStreamCancellationAndTimeouts(t *testing.T) {
	t.Run("cancel while stalled", func(t *testing.T) {
		closed := make(chan struct{}, 1)
		s := newStreamServer(t, stallAfter(closed, content("Part")))
		p, tr := streamProvider(s, Config{})
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var events []llm.StreamEvent
		for event := range p.Stream(ctx, chatReq("llama3.2:3b")) {
			events = append(events, event)
			if _, ok := event.(llm.TextDelta); ok {
				cancel()
			}
		}
		if len(events) != 2 {
			t.Fatalf("events %#v", events)
		}
		if f, ok := events[1].(llm.StreamFailed); !ok || !errors.Is(f.Err, context.Canceled) || f.PartialText != "Part" {
			t.Errorf("terminal %#v", events[1])
		}
		<-closed
		if tr.Open() != 0 {
			t.Error("open body")
		}
	})
	t.Run("idle timeout", func(t *testing.T) {
		closed := make(chan struct{}, 1)
		s := newStreamServer(t, stallAfter(closed, content("Part")))
		p, tr := streamProvider(s, Config{ChatTimeout: 50 * time.Millisecond})
		r := llmtest.Collect(t, p.Stream(context.Background(), chatReq("llama3.2:3b")))
		if f, ok := r.Terminal.(llm.StreamFailed); !ok || kindOf(t, f.Err) != llm.KindTimeout || f.PartialText != "Part" {
			t.Errorf("terminal %#v", r.Terminal)
		}
		<-closed
		if tr.Open() != 0 {
			t.Error("open body")
		}
	})
	t.Run("no limit on a long healthy stream", func(t *testing.T) {
		s := newStreamServer(t, func(w http.ResponseWriter, r *http.Request) {
			for range 6 { // 6 × 20 ms is longer than the 50 ms timeout
				ndjson(content("."))(w, r)
				time.Sleep(20 * time.Millisecond)
			}
			ndjson(finalLine)(w, r)
		})
		p, _ := streamProvider(s, Config{ChatTimeout: 50 * time.Millisecond})
		r := llmtest.Collect(t, p.Stream(context.Background(), chatReq("llama3.2:3b")))
		if _, ok := r.Terminal.(llm.StreamCompleted); !ok || r.Text() != "......" {
			t.Errorf("terminal %#v, text %q", r.Terminal, r.Text())
		}
	})
	t.Run("consumer stops early", func(t *testing.T) {
		closed := make(chan struct{}, 1)
		s := newStreamServer(t, stallAfter(closed, content("one"), content("two")))
		p, tr := streamProvider(s, Config{})
		for range p.Stream(context.Background(), chatReq("llama3.2:3b")) {
			break
		}
		<-closed
		if tr.Open() != 0 {
			t.Error("open body")
		}
	})
}
