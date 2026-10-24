package openai

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

	sdk "github.com/openai/openai-go/v3"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// fakeKey is a synthetic credential; tests never use a real API key.
const fakeKey = "sk-test-synthetic-key-0000"

// fakeAPI is an httptest OpenAI server that records every request.
type fakeAPI struct {
	*httptest.Server
	mu       sync.Mutex
	requests []recorded
}

type recorded struct {
	Method, Path, Auth string
	Body               map[string]any
}

// newFakeAPI serves routes keyed by "METHOD /path". Any other request fails
// the test.
func newFakeAPI(t *testing.T, routes map[string]http.HandlerFunc) *fakeAPI {
	t.Helper()
	f := &fakeAPI{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := recorded{Method: r.Method, Path: r.URL.Path, Auth: r.Header.Get("Authorization")}
		if data, _ := io.ReadAll(r.Body); len(data) > 0 {
			if err := json.Unmarshal(data, &rec.Body); err != nil {
				t.Errorf("request body is not a JSON object: %s", data)
			}
		}
		f.mu.Lock()
		f.requests = append(f.requests, rec)
		f.mu.Unlock()
		handler, ok := routes[r.Method+" "+r.URL.Path]
		if !ok {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		handler(w, r)
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeAPI) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

func (f *fakeAPI) last() recorded {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[len(f.requests)-1]
}

func (f *fakeAPI) provider(cfg Config) *Provider {
	cfg.APIKey = fakeKey
	cfg.BaseURL = f.URL + "/v1"
	cfg.HTTPClient = f.Client()
	return New(cfg)
}

// reply answers with a JSON body. Retry-After-Ms: 0 makes SDK retries
// immediate, so retry tests stay fast and deterministic.
func reply(status int, body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After-Ms", "0")
		w.WriteHeader(status)
		io.WriteString(w, body)
	}
}

func blockUntilCanceled(w http.ResponseWriter, r *http.Request) { <-r.Context().Done() }

func intp(n int) *int { return &n }

func userRequest(model, text string) llm.Request {
	return llm.Request{Model: model, Messages: []llm.Message{{Role: llm.RoleUser, Content: text}}}
}

const completed = `{
  "id": "resp_123", "object": "response", "status": "completed", "model": "gpt-6-luna-2026-09-01",
  "service_tier": "default",
  "output": [
    {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "SECRET SUMMARY"}],
     "content": [{"type": "reasoning_text", "text": "SECRET REASONING"}]},
    {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "Neural networks ", "annotations": []},
                 {"type": "output_text", "text": "learn from data.", "annotations": []}]}
  ],
  "usage": {"input_tokens": 20, "input_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 2},
            "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 12}, "total_tokens": 50}
}`

func message(text string) string {
	return `{"type":"message","role":"assistant","content":[{"type":"output_text","text":` + quote(text) + `}]}`
}

func quote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func kindOf(t *testing.T, err error) llm.ErrorKind {
	t.Helper()
	var e *llm.Error
	if !errors.As(err, &e) {
		t.Fatalf("error %v (%T) is not an *llm.Error", err, err)
	}
	return e.Kind
}

func TestChatRequest(t *testing.T) {
	f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": reply(200, completed)})
	if _, err := f.provider(Config{}).Chat(context.Background(), userRequest("ft:my-org:custom-7", "What is a neural network?")); err != nil {
		t.Fatal(err)
	}
	req := f.last()
	if req.Auth != "Bearer "+fakeKey {
		t.Error("request is not authenticated with the configured key")
	}
	want := map[string]any{
		"model": "ft:my-org:custom-7",
		"input": []any{map[string]any{"role": "user", "content": "What is a neural network?"}},
		"store": false,
	}
	if !reflect.DeepEqual(req.Body, want) {
		t.Errorf("request body = %v, want %v (no stream flag, store false)", req.Body, want)
	}
}

func TestChatCompletedResponse(t *testing.T) {
	f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": reply(200, completed)})
	resp, err := f.provider(Config{}).Chat(context.Background(), userRequest(DefaultModel, "Hi"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Neural networks learn from data." {
		t.Errorf("text = %q", resp.Text)
	}
	if strings.Contains(resp.Text, "SECRET") || resp.Reasoning != "" {
		t.Errorf("reasoning leaked: text=%q reasoning=%q", resp.Text, resp.Reasoning)
	}
	if resp.ID != "resp_123" || resp.Model != "gpt-6-luna-2026-09-01" || resp.ServiceTier != "default" || resp.IncompleteReason != "" {
		t.Errorf("metadata = %+v", resp)
	}
	wantUsage := &llm.Usage{
		InputTokens: intp(20), OutputTokens: intp(30), TotalTokens: intp(50),
		CachedInputTokens: intp(8), CacheWriteTokens: intp(2), ReasoningTokens: intp(12),
	}
	if !reflect.DeepEqual(resp.Usage, wantUsage) {
		t.Errorf("usage = %+v, want %+v", resp.Usage, wantUsage)
	}
}

func TestChatResponseStatuses(t *testing.T) {
	tests := []struct {
		name           string
		body           string
		wantText       string
		wantIncomplete string
		wantErr        string
	}{
		{"no usage reported", `{"status":"completed","output":[` + message("ok") + `]}`, "ok", "", ""},
		{"missing status", `{"output":[` + message("ok") + `]}`, "ok", "", ""},
		{"incomplete at token limit", `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[` + message("partial") + `]}`,
			"partial", "the max output token limit was reached", ""},
		{"incomplete by content filter", `{"status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}`,
			"", "the content filter stopped the response", ""},
		{"incomplete without details", `{"status":"incomplete","output":[]}`, "", "unknown reason", ""},
		{"failed", `{"status":"failed","error":{"code":"server_error","message":"The model crashed."},"output":[]}`,
			"", "", "OpenAI response failed: The model crashed."},
		{"failed without message", `{"status":"failed","output":[]}`, "", "", "OpenAI response failed: no details"},
		{"in progress", `{"status":"in_progress","output":[]}`, "", "", "OpenAI response did not complete (status 'in_progress')."},
		{"refusal", `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}]}`,
			"", "", "OpenAI refused the request: I can't help with that."},
		// A completed response may have no text (as in Python); only the
		// status decides failure, and reasoning never becomes the answer.
		{"reasoning only", `{"status":"completed","output":[{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"SECRET"}]}]}`,
			"", "", ""},
		{"empty text", `{"status":"completed","output":[` + message("") + `]}`, "", "", ""},
		{"no output", `{"status":"completed","output":[]}`, "", "", ""},
		{"text beside refusal", `{"status":"completed","output":[` + message("Answer") + `,{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"no"}]}]}`,
			"Answer", "", ""},
		{"tool call ignored", `{"status":"completed","output":[{"type":"function_call","name":"f","arguments":"{}","call_id":"c"},` + message("Text") + `]}`,
			"Text", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": reply(200, tt.body)})
			resp, err := f.provider(Config{}).Chat(context.Background(), userRequest(DefaultModel, "Hi"))
			if tt.wantErr != "" {
				if err == nil || kindOf(t, err) != llm.KindAPI || err.Error() != tt.wantErr {
					t.Fatalf("err = %v, want API error %q", err, tt.wantErr)
				}
				if resp.Text != "" {
					t.Errorf("failed request returned text %q", resp.Text)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if resp.Text != tt.wantText || resp.IncompleteReason != tt.wantIncomplete {
				t.Errorf("text, incomplete = %q, %q; want %q, %q", resp.Text, resp.IncompleteReason, tt.wantText, tt.wantIncomplete)
			}
		})
	}
}

func TestInvalidChatReplies(t *testing.T) {
	tests := map[string]http.HandlerFunc{
		"not JSON": reply(200, `{"status":`),
		"not JSON content type": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/html")
			io.WriteString(w, "<html>proxy login</html>")
		},
	}
	for name, handler := range tests {
		t.Run(name, func(t *testing.T) {
			f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": handler})
			_, err := f.provider(Config{}).Chat(context.Background(), userRequest(DefaultModel, "Hi"))
			if err == nil || kindOf(t, err) != llm.KindAPI || err.Error() != "Invalid response from OpenAI." {
				t.Fatalf("err = %v, want invalid-response API error", err)
			}
		})
	}
}

func TestDefaultModelMakesNoRequest(t *testing.T) {
	f := newFakeAPI(t, nil)
	got, err := f.provider(Config{}).DefaultModel(context.Background())
	if err != nil || got != "gpt-6-luna" {
		t.Errorf("DefaultModel() = %q, %v; want gpt-6-luna", got, err)
	}
	if n := f.count(); n != 0 {
		t.Errorf("%d requests, want 0", n)
	}
}

func TestMissingAPIKey(t *testing.T) {
	for _, key := range []string{"", "   "} {
		f := newFakeAPI(t, nil)
		p := New(Config{APIKey: key, BaseURL: f.URL + "/v1", HTTPClient: f.Client()})
		_, chatErr := p.Chat(context.Background(), userRequest(DefaultModel, "Hi"))
		_, listErr := p.ListModels(context.Background())
		for _, err := range []error{chatErr, listErr} {
			if !errors.Is(err, ErrMissingAPIKey) || kindOf(t, err) != llm.KindMissingAPIKey || llm.ExitCode(err) != 3 {
				t.Fatalf("err = %v, want missing API key", err)
			}
			if err.Error() != "OPENAI_API_KEY is not set. Export it in your environment and try again." {
				t.Errorf("message = %q", err)
			}
		}
		if n := f.count(); n != 0 {
			t.Errorf("%d requests without a key, want 0", n)
		}
	}
}

func TestStatusErrors(t *testing.T) {
	tests := []struct {
		name         string
		status       int
		body         string
		wantKind     llm.ErrorKind
		wantExit     int
		wantMessage  string
		wantAttempts int // 1 + retries
	}{
		{"401", 401, `{"error":{"message":"Incorrect API key provided: ` + fakeKey + `","type":"invalid_request_error","code":"invalid_api_key"}}`,
			llm.KindAuthentication, 4, "OpenAI authentication failed. Check that OPENAI_API_KEY is valid.", 1},
		{"403", 403, `{"error":{"message":"Project does not have access to model","type":"invalid_request_error"}}`,
			llm.KindAuthentication, 4, "OpenAI denied access (HTTP 403). Check that OPENAI_API_KEY is valid and allowed to use this model.", 1},
		{"429", 429, `{"error":{"message":"You exceeded your current quota","type":"insufficient_quota"}}`,
			llm.KindRateLimit, 5, "OpenAI rate limit or quota exceeded. Wait and try again, or check your plan and billing.", 3},
		{"400", 400, `{"error":{"message":"Invalid value for 'input'.","type":"invalid_request_error","param":"input"}}`,
			llm.KindAPI, 1, "OpenAI API error (HTTP 400): Invalid value for 'input'.", 1},
		{"404 unknown model", 404, `{"error":{"message":"The model ` + "`gpt-6-luna`" + ` does not exist or you do not have access to it.","type":"invalid_request_error","code":"model_not_found"}}`,
			llm.KindAPI, 1, "OpenAI API error (HTTP 404): The model `gpt-6-luna` does not exist or you do not have access to it.", 1},
		{"500", 500, `{"error":{"message":"The server had an error.","type":"server_error"}}`,
			llm.KindAPI, 1, "OpenAI API error (HTTP 500): The server had an error.", 3},
		{"503 without JSON body", 503, `upstream unavailable`, llm.KindAPI, 1, "OpenAI API error (HTTP 503): Service Unavailable", 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": reply(tt.status, tt.body)})
			_, err := f.provider(Config{}).Chat(context.Background(), userRequest(DefaultModel, "private prompt text"))
			if kindOf(t, err) != tt.wantKind || llm.ExitCode(err) != tt.wantExit {
				t.Errorf("err = %v (kind %v), want kind %v, exit %d", err, kindOf(t, err), tt.wantKind, tt.wantExit)
			}
			if err.Error() != tt.wantMessage {
				t.Errorf("message = %q, want %q", err, tt.wantMessage)
			}
			assertSafe(t, err.Error(), f.URL)
			var apiErr *sdk.Error
			if !errors.As(err, &apiErr) || apiErr.StatusCode != tt.status {
				t.Errorf("errors.As(*openai.Error) = %v", apiErr)
			}
			if n := f.count(); n != tt.wantAttempts {
				t.Errorf("%d attempts, want %d", n, tt.wantAttempts)
			}
		})
	}
}

// assertSafe checks that a user-visible message contains no credential,
// request content or endpoint.
func assertSafe(t *testing.T, message, baseURL string) {
	t.Helper()
	for _, secret := range []string{fakeKey, "sk-", "Bearer", "private prompt text", baseURL, "/v1/"} {
		if strings.Contains(message, secret) {
			t.Errorf("message %q exposes %q", message, secret)
		}
	}
}

func TestChatRetryPolicy(t *testing.T) {
	tests := []struct {
		name         string
		retries      *int
		status       int
		wantAttempts int
	}{
		{"default retries server errors", nil, 500, 1 + ChatMaxRetries},
		{"retries disabled", intp(0), 500, 1},
		{"client errors are not retried", nil, 400, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeAPI(t, map[string]http.HandlerFunc{
				"POST /v1/responses": reply(tt.status, `{"error":{"message":"x"}}`),
			})
			f.provider(Config{ChatMaxRetries: tt.retries}).Chat(context.Background(), userRequest(DefaultModel, "Hi"))
			if n := f.count(); n != tt.wantAttempts {
				t.Errorf("%d attempts, want %d", n, tt.wantAttempts)
			}
		})
	}
	t.Run("recovers after a transient error", func(t *testing.T) {
		var mu sync.Mutex
		calls := 0
		f := newFakeAPI(t, map[string]http.HandlerFunc{
			"POST /v1/responses": func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				calls++
				first := calls == 1
				mu.Unlock()
				if first {
					reply(503, `{"error":{"message":"busy"}}`)(w, r)
					return
				}
				reply(200, completed)(w, r)
			},
		})
		resp, err := f.provider(Config{}).Chat(context.Background(), userRequest(DefaultModel, "Hi"))
		if err != nil || resp.ID != "resp_123" || f.count() != 2 {
			t.Errorf("resp=%+v err=%v attempts=%d", resp, err, f.count())
		}
	})
}

func TestConnectionRefused(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	zero := 0
	p := New(Config{APIKey: fakeKey, BaseURL: url + "/v1", ChatMaxRetries: &zero})
	_, err := p.Chat(context.Background(), userRequest(DefaultModel, "private prompt text"))
	if kindOf(t, err) != llm.KindConnection || llm.ExitCode(err) != 7 {
		t.Fatalf("err = %v, want connection error", err)
	}
	if err.Error() != "Could not connect to OpenAI. Check your network connection." {
		t.Errorf("message = %q", err)
	}
	assertSafe(t, err.Error(), url)
}

func TestTimeoutsAreNotRetried(t *testing.T) {
	t.Run("chat attempt timeout", func(t *testing.T) {
		f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": blockUntilCanceled})
		_, err := f.provider(Config{ChatTimeout: 50 * time.Millisecond}).Chat(context.Background(), userRequest(DefaultModel, "Hi"))
		assertTimeout(t, err)
		if n := f.count(); n != 1 {
			t.Errorf("%d attempts, want 1", n)
		}
	})
	t.Run("caller deadline", func(t *testing.T) {
		f := newFakeAPI(t, map[string]http.HandlerFunc{"POST /v1/responses": blockUntilCanceled})
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		_, err := f.provider(Config{}).Chat(ctx, userRequest(DefaultModel, "Hi"))
		assertTimeout(t, err)
		if n := f.count(); n != 1 {
			t.Errorf("%d attempts, want 1", n)
		}
	})
}

func assertTimeout(t *testing.T, err error) {
	t.Helper()
	if kindOf(t, err) != llm.KindTimeout || llm.ExitCode(err) != 6 {
		t.Fatalf("err = %v, want timeout", err)
	}
	if err.Error() != "Request to OpenAI timed out." {
		t.Errorf("message = %q", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Error("timeout does not wrap context.DeadlineExceeded")
	}
}

func TestCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	f := newFakeAPI(t, map[string]http.HandlerFunc{
		"POST /v1/responses": func(w http.ResponseWriter, r *http.Request) {
			cancel()
			<-r.Context().Done()
		},
	})
	_, err := f.provider(Config{}).Chat(ctx, userRequest(DefaultModel, "Hi"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	var e *llm.Error
	if errors.As(err, &e) {
		t.Errorf("cancellation reported as provider error %v", e)
	}
	if n := f.count(); n != 1 {
		t.Errorf("%d attempts, want 1 (no retry after cancel)", n)
	}
}

func TestListModels(t *testing.T) {
	const models = `{"object":"list","data":[
	  {"id":"gpt-6-luna","object":"model","created":1,"owned_by":"openai"},
	  {"id":"text-embedding-3-large","object":"model"},
	  {"id":"gpt-4o-mini-tts","object":"model"},
	  {"id":"whisper-1","object":"model"},
	  {"id":"dall-e-3","object":"model"},
	  {"id":"omni-moderation-latest","object":"model"},
	  {"id":"gpt-4o-transcribe","object":"model"},
	  {"id":"gpt-realtime","object":"model"},
	  {"id":"gpt-audio","object":"model"},
	  {"id":"gpt-image-1","object":"model"},
	  {"id":"gpt-5","object":"model"},
	  {"id":"gpt-5","object":"model"},
	  {"id":"","object":"model"},
	  {"id":"ft:gpt-4.1:org::abc","object":"model"}
	]}`
	f := newFakeAPI(t, map[string]http.HandlerFunc{"GET /v1/models": reply(200, models)})
	p := f.provider(Config{})
	got, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, m := range got {
		ids = append(ids, m.ID)
	}
	want := []string{"ft:gpt-4.1:org::abc", "gpt-5", "gpt-6-luna"}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("models = %v, want %v", ids, want)
	}
	if f.last().Auth != "Bearer "+fakeKey {
		t.Error("models request is not authenticated with the configured key")
	}
	if _, err := p.ListModels(context.Background()); err != nil || f.count() != 1 {
		t.Errorf("second ListModels: err=%v requests=%d, want cached", err, f.count())
	}
}

func TestListModelsEmptyAndMalformed(t *testing.T) {
	tests := []struct {
		name, body string
		wantErr    bool
	}{
		{"empty list", `{"object":"list","data":[]}`, false},
		{"no data field", `{"object":"list"}`, false},
		{"data not a list", `{"object":"list","data":"oops"}`, false},
		{"not JSON", `{"data":[`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeAPI(t, map[string]http.HandlerFunc{"GET /v1/models": reply(200, tt.body)})
			got, err := f.provider(Config{}).ListModels(context.Background())
			if tt.wantErr {
				if err == nil || kindOf(t, err) != llm.KindAPI {
					t.Fatalf("err = %v, want API error", err)
				}
				return
			}
			if err != nil || len(got) != 0 {
				t.Errorf("got %v, %v; want no models", got, err)
			}
		})
	}
}

func TestListModelsIsNeverRetried(t *testing.T) {
	t.Run("timeout", func(t *testing.T) {
		f := newFakeAPI(t, map[string]http.HandlerFunc{"GET /v1/models": blockUntilCanceled})
		_, err := f.provider(Config{DiscoveryTimeout: 50 * time.Millisecond}).ListModels(context.Background())
		assertTimeout(t, err)
		if n := f.count(); n != 1 {
			t.Errorf("%d attempts, want 1", n)
		}
	})
	for _, status := range []int{429, 500, 503} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			f := newFakeAPI(t, map[string]http.HandlerFunc{"GET /v1/models": reply(status, `{"error":{"message":"x"}}`)})
			_, err := f.provider(Config{}).ListModels(context.Background())
			if err == nil {
				t.Fatal("no error")
			}
			if n := f.count(); n != 1 {
				t.Errorf("%d attempts, want 1", n)
			}
		})
	}
	if DiscoveryTimeout != 5*time.Second {
		t.Errorf("DiscoveryTimeout = %v, want 5s (spec section 12)", DiscoveryTimeout)
	}
}

func TestCheckAvailability(t *testing.T) {
	f := newFakeAPI(t, map[string]http.HandlerFunc{"GET /v1/models": reply(200, `{"object":"list","data":[{"id":"gpt-6-luna","object":"model"}]}`)})
	if err := f.provider(Config{}).CheckAvailability(context.Background()); err != nil {
		t.Error(err)
	}
	if err := New(Config{}).CheckAvailability(context.Background()); !errors.Is(err, ErrMissingAPIKey) {
		t.Errorf("without a key: %v", err)
	}
	f = newFakeAPI(t, map[string]http.HandlerFunc{"GET /v1/models": reply(500, `{"error":{"message":"down"}}`)})
	if err := f.provider(Config{}).CheckAvailability(context.Background()); err == nil || strings.Contains(err.Error(), fakeKey) {
		t.Errorf("server error: %v", err)
	}
}
