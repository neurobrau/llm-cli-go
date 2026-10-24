package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// fakeServer is an httptest Ollama server that records every request.
type fakeServer struct {
	*httptest.Server
	mu       sync.Mutex
	requests []recorded
}

type recorded struct {
	Method, Path, ContentType string
	Body                      map[string]any
}

// newFakeServer serves routes keyed by "METHOD /path". Any other request
// fails the test.
func newFakeServer(t *testing.T, routes map[string]http.HandlerFunc) *fakeServer {
	t.Helper()
	s := &fakeServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := recorded{Method: r.Method, Path: r.URL.Path, ContentType: r.Header.Get("Content-Type")}
		if data, _ := io.ReadAll(r.Body); len(data) > 0 {
			if err := json.Unmarshal(data, &rec.Body); err != nil {
				t.Errorf("request body is not a JSON object: %s", data)
			}
		}
		s.mu.Lock()
		s.requests = append(s.requests, rec)
		s.mu.Unlock()
		handler, ok := routes[r.Method+" "+r.URL.Path]
		if !ok {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		handler(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *fakeServer) requestsTo(path string) []recorded {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []recorded
	for _, r := range s.requests {
		if r.Path == path {
			out = append(out, r)
		}
	}
	return out
}

func (s *fakeServer) provider() *Provider {
	return New(Config{BaseURL: s.URL, HTTPClient: s.Client()})
}

func reply(status int, body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		io.WriteString(w, body)
	}
}

// blockUntilCanceled holds the request open until the client gives up.
func blockUntilCanceled(w http.ResponseWriter, r *http.Request) { <-r.Context().Done() }

func tagsJSON(entries ...string) string {
	return `{"models":[` + strings.Join(entries, ",") + `]}`
}

const (
	localQwen  = `{"name":"qwen3.5:4b","model":"qwen3.5:4b","capabilities":["completion","thinking"],"details":{"parameter_size":"4B","quantization_level":"Q4_K_M","context_length":32768}}`
	localLlama = `{"name":"llama3.2:3b","model":"llama3.2:3b","capabilities":["completion","tools"]}`
	cloudTag   = `{"name":"glm-4.7:cloud","model":"glm-4.7:cloud","capabilities":["completion"]}`
	cloudDash  = `{"name":"gemma4:31b-cloud","model":"gemma4:31b-cloud"}`
	remoteHost = `{"name":"remote-a:latest","model":"remote-a:latest","remote_host":"https://ollama.com:443","capabilities":["completion"]}`
	remoteMdl  = `{"name":"remote-b:latest","model":"remote-b:latest","remote_model":"b","capabilities":["completion"]}`
	embedding  = `{"name":"nomic-embed-text:latest","model":"nomic-embed-text:latest","capabilities":["embedding"]}`
	oldOllama  = `{"name":"mistral:7b","model":"mistral:7b"}`
)

const okChat = `{"model":"qwen3.5:4b","message":{"role":"assistant","content":"Hello!"},"done":true,"done_reason":"stop",
"prompt_eval_count":12,"eval_count":5,"total_duration":2500000000,"load_duration":1000000000,"prompt_eval_duration":300000000,"eval_duration":1200000000}`

func kindOf(t *testing.T, err error) llm.ErrorKind {
	t.Helper()
	var e *llm.Error
	if !errors.As(err, &e) {
		t.Fatalf("error %v (%T) is not an *llm.Error", err, err)
	}
	return e.Kind
}

func TestResolveBaseURL(t *testing.T) {
	tests := []struct{ in, want string }{
		{"", DefaultBaseURL},
		{"   ", DefaultBaseURL},
		{"localhost", "http://localhost:11434"},
		{"example.lan:8080", "http://example.lan:8080"},
		{"0.0.0.0", "http://127.0.0.1:11434"},
		{"0.0.0.0:9999", "http://127.0.0.1:9999"},
		{"[::]:11434", "http://127.0.0.1:11434"},
		{"[::1]:1234", "http://[::1]:1234"},
		{"https://ollama.example.com", "https://ollama.example.com:11434"},
		{"http://host:1/prefix/", "http://host:1/prefix"},
		{":5000", "http://127.0.0.1:5000"},
	}
	for _, tt := range tests {
		got, err := ResolveBaseURL(tt.in)
		if err != nil || got != tt.want {
			t.Errorf("ResolveBaseURL(%q) = %q, %v; want %q", tt.in, got, err, tt.want)
		}
	}
	if _, err := ResolveBaseURL("host:notaport"); err == nil {
		t.Error("ResolveBaseURL accepted an invalid port")
	}
}

func TestListModelsFiltersRemoteAndNonChatModels(t *testing.T) {
	s := newFakeServer(t, map[string]http.HandlerFunc{
		"GET /api/tags": reply(200, tagsJSON(cloudTag, localQwen, cloudDash, remoteHost, embedding, remoteMdl, oldOllama, localLlama)),
	})
	models, err := s.provider().ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, m := range models {
		ids = append(ids, m.ID)
	}
	want := []string{"qwen3.5:4b", "mistral:7b", "llama3.2:3b"}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("models = %v, want %v (tags order, local chat models only)", ids, want)
	}
	if models[0].Details != "4B Q4_K_M" || models[0].ContextLength != 32768 {
		t.Errorf("details = %q, context = %d", models[0].Details, models[0].ContextLength)
	}
	if models[1].Capabilities != nil {
		t.Errorf("unreported capabilities should stay nil, got %v", models[1].Capabilities)
	}
	if n := len(s.requestsTo("/api/show")); n != 0 {
		t.Errorf("listing made %d /api/show requests, want 0", n)
	}
}

func TestDefaultModelIsFirstSelectableInTagsOrder(t *testing.T) {
	s := newFakeServer(t, map[string]http.HandlerFunc{
		"GET /api/tags": reply(200, tagsJSON(cloudTag, embedding, localLlama, localQwen)),
	})
	got, err := s.provider().DefaultModel(context.Background())
	if err != nil || got != "llama3.2:3b" {
		t.Errorf("DefaultModel() = %q, %v; want llama3.2:3b", got, err)
	}
}

func TestDefaultModelWithoutLocalModels(t *testing.T) {
	for name, body := range map[string]string{
		"empty":      tagsJSON(),
		"cloud only": tagsJSON(cloudTag, cloudDash, remoteHost),
	} {
		t.Run(name, func(t *testing.T) {
			s := newFakeServer(t, map[string]http.HandlerFunc{"GET /api/tags": reply(200, body)})
			_, err := s.provider().DefaultModel(context.Background())
			if !errors.Is(err, ErrNoLocalModels) || kindOf(t, err) != llm.KindAPI {
				t.Fatalf("err = %v, want ErrNoLocalModels", err)
			}
			if !strings.Contains(err.Error(), "ollama pull <model>") {
				t.Errorf("message not actionable: %q", err)
			}
		})
	}
}

func TestChatRequestAndResponse(t *testing.T) {
	s := newFakeServer(t, map[string]http.HandlerFunc{
		"GET /api/tags":  reply(200, tagsJSON(localLlama, localQwen)),
		"POST /api/chat": reply(200, okChat),
	})
	p := s.provider()
	ctx := context.Background()
	model, err := p.DefaultModel(ctx)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Chat(ctx, llm.Request{Model: model, Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}}})
	if err != nil {
		t.Fatal(err)
	}

	chats := s.requestsTo("/api/chat")
	if len(chats) != 1 {
		t.Fatalf("%d chat requests, want 1", len(chats))
	}
	if chats[0].Method != http.MethodPost || chats[0].ContentType != "application/json" {
		t.Errorf("chat request %s with Content-Type %q", chats[0].Method, chats[0].ContentType)
	}
	wantBody := map[string]any{
		"model":    "llama3.2:3b",
		"messages": []any{map[string]any{"role": "user", "content": "Hi"}},
		"stream":   false,
	}
	if !reflect.DeepEqual(chats[0].Body, wantBody) {
		t.Errorf("chat body = %v, want %v", chats[0].Body, wantBody)
	}
	if n := len(s.requestsTo("/api/tags")); n != 1 {
		t.Errorf("%d /api/tags requests, want 1 (cached)", n)
	}

	if resp.Text != "Hello!" || resp.Reasoning != "" || resp.Model != "qwen3.5:4b" || resp.DoneReason != "stop" {
		t.Errorf("response = %+v", resp)
	}
	if resp.Usage == nil || *resp.Usage.InputTokens != 12 || *resp.Usage.OutputTokens != 5 || *resp.Usage.TotalTokens != 17 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	wantTiming := llm.Timing{Total: 2500 * time.Millisecond, Load: time.Second, PromptEval: 300 * time.Millisecond, Eval: 1200 * time.Millisecond}
	if resp.Timing != wantTiming {
		t.Errorf("timing = %+v, want %+v", resp.Timing, wantTiming)
	}
}

func TestChatThinkParameter(t *testing.T) {
	tests := []struct {
		name      string
		entry     string
		model     string
		show      string // /api/show reply; "" means it must not be requested
		wantThink any    // nil: no "think" field
	}{
		{"thinking capability in tags", localQwen, "qwen3.5:4b", "", false},
		{"no thinking capability in tags", localLlama, "llama3.2:3b", "", nil},
		{"capabilities from /api/show: thinking", oldOllama, "mistral:7b", `{"capabilities":["completion","thinking"]}`, false},
		{"capabilities from /api/show: none", oldOllama, "mistral:7b", `{"capabilities":["completion"]}`, nil},
		{"/api/show without capabilities", oldOllama, "mistral:7b", `{}`, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			routes := map[string]http.HandlerFunc{
				"GET /api/tags":  reply(200, tagsJSON(tt.entry)),
				"POST /api/chat": reply(200, okChat),
			}
			if tt.show != "" {
				routes["POST /api/show"] = reply(200, tt.show)
			}
			s := newFakeServer(t, routes)
			p := s.provider()
			req := llm.Request{Model: tt.model, Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}}}
			for range 2 { // the second chat must reuse cached discovery
				if _, err := p.Chat(context.Background(), req); err != nil {
					t.Fatal(err)
				}
			}
			for _, chat := range s.requestsTo("/api/chat") {
				think, present := chat.Body["think"]
				if tt.wantThink == nil && present {
					t.Errorf("think = %v sent, want no think field", think)
				}
				if tt.wantThink != nil && think != tt.wantThink {
					t.Errorf("think = %v (present %v), want %v", think, present, tt.wantThink)
				}
			}
			shows := s.requestsTo("/api/show")
			wantShows := 0
			if tt.show != "" {
				wantShows = 1
			}
			if len(shows) != wantShows {
				t.Fatalf("%d /api/show requests, want %d", len(shows), wantShows)
			}
			if wantShows == 1 && !reflect.DeepEqual(shows[0].Body, map[string]any{"model": tt.model}) {
				t.Errorf("/api/show body = %v", shows[0].Body)
			}
		})
	}
}

func TestChatRefusesMissingAndCloudModels(t *testing.T) {
	tests := []struct {
		model      string
		sentinel   error
		wantText   string
		wantTagsNo int
	}{
		{"absent:1b", ErrModelNotInstalled, "Ollama model 'absent:1b' is not installed locally. Install it yourself with: ollama pull absent:1b", 2},
		{"glm-4.7:cloud", ErrCloudModel, "'glm-4.7:cloud' is an Ollama cloud model", 1},
		{"gemma4:31b-cloud", ErrCloudModel, "cloud model", 1},
		{"remote-a:latest", ErrCloudModel, "cloud model", 1},
		{"remote-b:latest", ErrCloudModel, "cloud model", 1},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			// No /api/show or /api/chat route: any such request fails the test.
			s := newFakeServer(t, map[string]http.HandlerFunc{
				"GET /api/tags": reply(200, tagsJSON(localQwen, cloudTag, cloudDash, remoteHost, remoteMdl)),
			})
			_, err := s.provider().Chat(context.Background(), llm.Request{Model: tt.model})
			if !errors.Is(err, tt.sentinel) || kindOf(t, err) != llm.KindAPI || llm.ExitCode(err) != 1 {
				t.Fatalf("err = %v, want %v", err, tt.sentinel)
			}
			if !strings.Contains(err.Error(), tt.wantText) {
				t.Errorf("message %q does not contain %q", err, tt.wantText)
			}
			if n := len(s.requestsTo("/api/tags")); n != tt.wantTagsNo {
				t.Errorf("%d /api/tags requests, want %d", n, tt.wantTagsNo)
			}
		})
	}
}

func TestChatFindsModelInstalledSinceDiscovery(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	s := newFakeServer(t, map[string]http.HandlerFunc{
		"GET /api/tags": func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			calls++
			body := tagsJSON(localLlama)
			if calls > 1 {
				body = tagsJSON(localQwen, localLlama)
			}
			mu.Unlock()
			reply(200, body)(w, r)
		},
		"POST /api/chat": reply(200, okChat),
	})
	p := s.provider()
	if _, err := p.ListModels(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Chat(context.Background(), llm.Request{Model: "qwen3.5:4b"}); err != nil {
		t.Fatalf("Chat after refresh: %v", err)
	}
}

func TestChatResponseParsing(t *testing.T) {
	tests := []struct {
		name, body         string
		wantText, wantReas string
		wantUsage          *[3]int // input, output, total; nil: no usage
	}{
		{"thinking field", `{"message":{"content":"Answer","thinking":"hmm"}}`, "Answer", "hmm", nil},
		{"leading think block", `{"message":{"content":"  <think>\nplan\n</think>\n\nAnswer"}}`, "Answer", "plan", nil},
		{"thinking field wins over block", `{"message":{"content":"<think>x</think>Answer","thinking":"y"}}`, "Answer", "y", nil},
		{"unclosed think block", `{"message":{"content":"<think>never finished"}}`, "", "never finished", nil},
		{"think block not leading", `{"message":{"content":"A <think>b</think>"}}`, "A <think>b</think>", "", nil},
		{"empty content", `{"message":{"content":""}}`, "", "", nil},
		{"output tokens only", `{"message":{"content":"x"},"eval_count":3}`, "x", "", &[3]int{-1, 3, -1}},
		{"non-integer counts ignored", `{"message":{"content":"x"},"eval_count":1.5,"prompt_eval_count":"7"}`, "x", "", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := parseChatResponse([]byte(tt.body))
			if err != nil {
				t.Fatal(err)
			}
			if resp.Text != tt.wantText || resp.Reasoning != tt.wantReas {
				t.Errorf("text, reasoning = %q, %q; want %q, %q", resp.Text, resp.Reasoning, tt.wantText, tt.wantReas)
			}
			if got := usageTriple(resp.Usage); !reflect.DeepEqual(got, tt.wantUsage) {
				t.Errorf("usage = %v, want %v", got, tt.wantUsage)
			}
		})
	}
}

func usageTriple(u *llm.Usage) *[3]int {
	if u == nil {
		return nil
	}
	val := func(p *int) int {
		if p == nil {
			return -1
		}
		return *p
	}
	return &[3]int{val(u.InputTokens), val(u.OutputTokens), val(u.TotalTokens)}
}

func TestInvalidResponses(t *testing.T) {
	tests := []struct {
		name, tags, chat, want string
	}{
		{"tags not JSON", `<html>`, okChat, "Invalid response from Ollama: expected an object with a 'models' list."},
		{"tags without list", `{"models":null}`, okChat, "expected an object with a 'models' list"},
		{"tags entry without name", `{"models":[{"size":1}]}`, okChat, "model entry has no name"},
		{"tags entry not object", `{"models":[42]}`, okChat, "model entry is not an object"},
		{"chat not JSON", tagsJSON(localLlama), `{"message":`, "Invalid response from Ollama: expected a JSON object."},
		{"chat without content", tagsJSON(localLlama), `{"message":{"role":"assistant"}}`, "missing 'message.content'"},
		{"chat without message", tagsJSON(localLlama), `{"done":true}`, "missing 'message.content'"},
		{"chat error payload", tagsJSON(localLlama), `{"error":"model crashed"}`, "Ollama error: model crashed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newFakeServer(t, map[string]http.HandlerFunc{
				"GET /api/tags":  reply(200, tt.tags),
				"POST /api/chat": reply(200, tt.chat),
			})
			_, err := s.provider().Chat(context.Background(), llm.Request{Model: "llama3.2:3b"})
			if err == nil || kindOf(t, err) != llm.KindAPI {
				t.Fatalf("err = %v, want an API error", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("message %q does not contain %q", err, tt.want)
			}
		})
	}
}

func TestHTTPErrors(t *testing.T) {
	tests := []struct {
		name       string
		tags, chat http.HandlerFunc
		want       string
		wantStatus int
	}{
		{"tags server error", reply(500, `{"error":"boom"}`), nil, "Ollama API error (HTTP 500): boom", 500},
		{"chat error text", reply(200, tagsJSON(localLlama)), reply(400, `{"error":"bad request"}`), "Ollama API error (HTTP 400): bad request", 400},
		{"chat non-JSON error", reply(200, tagsJSON(localLlama)), reply(502, "Bad Gateway"), "Ollama API error (HTTP 502): Bad Gateway", 502},
		{"chat empty error body", reply(200, tagsJSON(localLlama)), reply(503, ""), "Ollama API error (HTTP 503): no details", 503},
		{"chat model not found", reply(200, tagsJSON(localLlama)), reply(404, `{"error":"model 'llama3.2:3b' not found"}`),
			"Ollama model 'llama3.2:3b' was not found: model 'llama3.2:3b' not found. Install it yourself with: ollama pull llama3.2:3b", 404},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			routes := map[string]http.HandlerFunc{"GET /api/tags": tt.tags}
			if tt.chat != nil {
				routes["POST /api/chat"] = tt.chat
			}
			s := newFakeServer(t, routes)
			_, err := s.provider().Chat(context.Background(), llm.Request{Model: "llama3.2:3b"})
			if err == nil || kindOf(t, err) != llm.KindAPI || llm.ExitCode(err) != 1 {
				t.Fatalf("err = %v, want an API error", err)
			}
			if err.Error() != tt.want {
				t.Errorf("message = %q, want %q", err, tt.want)
			}
			var httpErr *HTTPError
			if !errors.As(err, &httpErr) || httpErr.StatusCode != tt.wantStatus {
				t.Errorf("errors.As HTTPError = %v", httpErr)
			}
		})
	}
}

func TestConnectionRefused(t *testing.T) {
	s := httptest.NewServer(http.NotFoundHandler())
	url := s.URL
	s.Close() // nothing listens on this port any more

	p := New(Config{BaseURL: url})
	_, err := p.DefaultModel(context.Background())
	if kindOf(t, err) != llm.KindConnection || llm.ExitCode(err) != 7 {
		t.Fatalf("err = %v, want a connection error", err)
	}
	want := "Could not connect to Ollama at " + url + ". Is the Ollama service running?"
	if err.Error() != want {
		t.Errorf("message = %q, want %q", err, want)
	}
}

func TestTimeouts(t *testing.T) {
	t.Run("discovery", func(t *testing.T) {
		s := newFakeServer(t, map[string]http.HandlerFunc{"GET /api/tags": blockUntilCanceled})
		p := New(Config{BaseURL: s.URL, HTTPClient: s.Client(), DiscoveryTimeout: 50 * time.Millisecond})
		_, err := p.ListModels(context.Background())
		assertTimeout(t, err)
	})
	t.Run("chat", func(t *testing.T) {
		s := newFakeServer(t, map[string]http.HandlerFunc{
			"GET /api/tags":  reply(200, tagsJSON(localLlama)),
			"POST /api/chat": blockUntilCanceled,
		})
		p := New(Config{BaseURL: s.URL, HTTPClient: s.Client(), ChatTimeout: 50 * time.Millisecond})
		_, err := p.Chat(context.Background(), llm.Request{Model: "llama3.2:3b"})
		assertTimeout(t, err)
	})
	t.Run("caller deadline", func(t *testing.T) {
		s := newFakeServer(t, map[string]http.HandlerFunc{"GET /api/tags": blockUntilCanceled})
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		_, err := s.provider().ListModels(ctx)
		assertTimeout(t, err)
	})
}

func assertTimeout(t *testing.T, err error) {
	t.Helper()
	if kindOf(t, err) != llm.KindTimeout || llm.ExitCode(err) != 6 {
		t.Fatalf("err = %v, want a timeout error", err)
	}
	if err.Error() != "Request to Ollama timed out." {
		t.Errorf("message = %q", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Error("timeout error does not wrap context.DeadlineExceeded")
	}
}

func TestCancellation(t *testing.T) {
	started := make(chan struct{})
	s := newFakeServer(t, map[string]http.HandlerFunc{
		"GET /api/tags": reply(200, tagsJSON(localLlama)),
		"POST /api/chat": func(w http.ResponseWriter, r *http.Request) {
			close(started)
			<-r.Context().Done()
		},
	})
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-started
		cancel()
	}()
	_, err := s.provider().Chat(ctx, llm.Request{Model: "llama3.2:3b"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	var e *llm.Error
	if errors.As(err, &e) {
		t.Errorf("cancellation reported as provider error %v", e)
	}
}

func TestCheckAvailability(t *testing.T) {
	s := newFakeServer(t, map[string]http.HandlerFunc{"GET /api/tags": reply(200, tagsJSON(cloudTag, localQwen))})
	p := s.provider()
	if err := p.CheckAvailability(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := p.CheckAvailability(context.Background()); err != nil || len(s.requestsTo("/api/tags")) != 2 {
		t.Errorf("err %v; %d /api/tags requests, want 2 (the check reloads)", err, len(s.requestsTo("/api/tags")))
	}

	cloudOnly := newFakeServer(t, map[string]http.HandlerFunc{"GET /api/tags": reply(200, tagsJSON(cloudTag, cloudDash))})
	if err := cloudOnly.provider().CheckAvailability(context.Background()); !errors.Is(err, ErrNoLocalModels) {
		t.Errorf("cloud only: %v", err)
	}
	down := New(Config{BaseURL: "http://127.0.0.1:1"})
	if err := down.CheckAvailability(context.Background()); kindOf(t, err) != llm.KindConnection {
		t.Errorf("offline: %v", err)
	}
}
