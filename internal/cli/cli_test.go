package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// isolateEnv makes sure a test can neither reach a real Ollama or OpenAI
// server nor see credentials or real user state directories.
func isolateEnv(t *testing.T) {
	t.Helper()
	t.Setenv("OLLAMA_HOST", "127.0.0.1:1")               // nothing listens on port 1
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1") // read by the OpenAI SDK
	for _, key := range []string{
		"OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
		"OPENAI_WEBHOOK_SECRET", "OPENAI_CUSTOM_HEADERS",
	} {
		t.Setenv(key, "")
		os.Unsetenv(key) // restored by t.Setenv's cleanup
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", home+"/config")
	t.Setenv("XDG_STATE_HOME", home+"/state")
}

func run(t *testing.T, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	return runCtx(t, context.Background(), args...)
}

func runCtx(t *testing.T, ctx context.Context, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errOut bytes.Buffer
	// Empty input: an interactive chat ends at once (EOF).
	code = Run(ctx, args, Streams{Stdin: strings.NewReader(""), Stdout: &out, Stderr: &errOut})
	return code, out.String(), errOut.String()
}

func TestHelp(t *testing.T) {
	isolateEnv(t)
	for _, arg := range []string{"--help", "-h", "-help"} {
		t.Run(arg, func(t *testing.T) {
			code, stdout, stderr := run(t, arg)
			if code != ExitOK {
				t.Errorf("exit code = %d, want %d", code, ExitOK)
			}
			if !strings.HasPrefix(stdout, "usage: neurobrau ") {
				t.Errorf("stdout does not start with usage line:\n%s", stdout)
			}
			for _, want := range []string{"--help", "--version", "--provider", "--model", "openai or ollama (default: openai;", "--plain", "--no-history", "omit to",
				"gpt-6-luna", "OPENAI_API_KEY", "OLLAMA_HOST", "Not implemented yet:"} {
				if !strings.Contains(stdout, want) {
					t.Errorf("help text missing %q:\n%s", want, stdout)
				}
			}
			if stderr != "" {
				t.Errorf("stderr = %q, want empty", stderr)
			}
		})
	}
}

func TestHelpTakesPrecedenceOverLaterArguments(t *testing.T) {
	isolateEnv(t)
	code, stdout, stderr := run(t, "--provider", "ollama", "Hi", "--help")
	if code != ExitOK || !strings.HasPrefix(stdout, "usage: ") || stderr != "" {
		t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

func TestVersion(t *testing.T) {
	isolateEnv(t)
	code, stdout, stderr := run(t, "--version")
	if code != ExitOK {
		t.Errorf("exit code = %d, want %d", code, ExitOK)
	}
	if want := "NeuroBräu Go CLI 0.1.0-dev\n"; stdout != want {
		t.Errorf("stdout = %q, want %q", stdout, want)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty", stderr)
	}
}

func TestUsageErrors(t *testing.T) {
	isolateEnv(t)
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"unknown long flag", []string{"--not-a-real-flag"}, "not-a-real-flag"},
		{"unknown short flag", []string{"-x"}, "-x"},
		{"empty prompt", []string{""}, "prompt must not be empty"},
		{"blank prompt", []string{"--provider", "ollama", "  "}, "prompt must not be empty"},
		{"positional after version", []string{"--version", "extra"}, `unexpected argument: "extra"`},
		{"malformed boolean value", []string{"--version=maybe"}, "version"},
		{"invalid provider", []string{"--provider", "anthropic", "Hi"}, `argument --provider: invalid choice: "anthropic" (choose from openai, ollama)`},
		{"invalid provider without prompt", []string{"--provider", "anthropic"}, "invalid choice"},
		{"two prompts", []string{"--provider", "ollama", "a", "b"}, `unexpected argument: "b"`},
		{"model without value", []string{"--provider", "ollama", "--model"}, "flag needs an argument: -model"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, stdout, stderr := run(t, tt.args...)
			if code != ExitUsage {
				t.Errorf("exit code = %d, want %d", code, ExitUsage)
			}
			if stdout != "" {
				t.Errorf("stdout = %q, want empty", stdout)
			}
			if !strings.HasPrefix(stderr, "neurobrau: error: ") {
				t.Errorf("stderr = %q, want prefix %q", stderr, "neurobrau: error: ")
			}
			if !strings.Contains(stderr, tt.wantErr) {
				t.Errorf("stderr = %q, want it to contain %q", stderr, tt.wantErr)
			}
			if strings.Count(stderr, "\n") != 1 || !strings.HasSuffix(stderr, "\n") {
				t.Errorf("stderr should be exactly one line: %q", stderr)
			}
		})
	}
}

func TestInvalidOllamaHost(t *testing.T) {
	isolateEnv(t)
	t.Setenv("OLLAMA_HOST", "localhost:notaport")
	code, stdout, stderr := run(t, "--provider", "ollama", "Hi")
	if code != ExitUsage || stdout != "" || !strings.HasPrefix(stderr, `neurobrau: error: invalid OLLAMA_HOST value "localhost:notaport"`) {
		t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

// fakeOllama starts an httptest Ollama server, points OLLAMA_HOST at it and
// records the /api/chat request bodies.
type fakeOllama struct {
	mu    sync.Mutex
	chats []map[string]any
}

func startFakeOllama(t *testing.T, tags string, chat http.HandlerFunc) *fakeOllama {
	t.Helper()
	return startFakeOllamaWithShow(t, tags, nil, chat)
}

// startFakeOllamaWithShow also answers /api/show with shows[model]; a model
// without an entry fails the test.
func startFakeOllamaWithShow(t *testing.T, tags string, shows map[string]string, chat http.HandlerFunc) *fakeOllama {
	t.Helper()
	isolateEnv(t)
	f := &fakeOllama{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method + " " + r.URL.Path {
		case "GET /api/tags":
			io.WriteString(w, tags)
		case "POST /api/show":
			var body struct{ Model string }
			json.NewDecoder(r.Body).Decode(&body)
			show, ok := shows[body.Model]
			if !ok {
				t.Errorf("unexpected /api/show for %q", body.Model)
			}
			io.WriteString(w, show)
		case "POST /api/chat":
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			f.mu.Lock()
			f.chats = append(f.chats, body)
			f.mu.Unlock()
			chat(w, r)
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)
	return f
}

func (f *fakeOllama) chatModels() []any {
	f.mu.Lock()
	defer f.mu.Unlock()
	var models []any
	for _, c := range f.chats {
		models = append(models, c["model"])
	}
	return models
}

func respond(status int, body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		io.WriteString(w, body)
	}
}

const (
	tags      = `{"models":[{"model":"glm-4.7:cloud","capabilities":["completion"]},{"model":"qwen3.5:4b","capabilities":["completion","thinking"]},{"model":"llama3.2:3b","capabilities":["completion"]}]}`
	emptyTags = `{"models":[]}`
	answer    = `{"model":"qwen3.5:4b","message":{"role":"assistant","content":"<think>secret plan</think>Neural networks learn.","thinking":"more thoughts"},"done":true,"prompt_eval_count":10,"eval_count":4}`
)

func TestOllamaOneShot(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantModel string
	}{
		{"explicit model", []string{"--provider", "ollama", "--model", "llama3.2:3b", "Explain neural networks"}, "llama3.2:3b"},
		{"default model skips cloud models", []string{"--provider", "ollama", "Explain neural networks"}, "qwen3.5:4b"},
		{"prompt before flags", []string{"Explain neural networks", "--provider", "ollama"}, "qwen3.5:4b"},
		{"prompt after terminator", []string{"--provider=ollama", "--", "Explain neural networks"}, "qwen3.5:4b"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := startFakeOllama(t, tags, respond(200, answer))
			code, stdout, stderr := run(t, tt.args...)
			if code != ExitOK || stderr != "" {
				t.Fatalf("got code=%d stderr=%q", code, stderr)
			}
			if want := "Neural networks learn.\n"; stdout != want {
				t.Errorf("stdout = %q, want %q (no reasoning)", stdout, want)
			}
			if got := f.chatModels(); len(got) != 1 || got[0] != tt.wantModel {
				t.Errorf("chat requests for models %v, want [%s]", got, tt.wantModel)
			}
		})
	}
}

func TestOllamaErrors(t *testing.T) {
	tests := []struct {
		name     string
		tags     string
		chat     http.HandlerFunc
		args     []string
		wantCode int
		wantErr  string
	}{
		{"missing model", tags, nil, []string{"--provider", "ollama", "--model", "absent:1b", "Hi"}, 1,
			"neurobrau: error: Ollama model 'absent:1b' is not installed locally. Install it yourself with: ollama pull absent:1b\n"},
		{"cloud model", tags, nil, []string{"--provider", "ollama", "--model", "glm-4.7:cloud", "Hi"}, 1,
			"neurobrau: error: 'glm-4.7:cloud' is an Ollama cloud model that runs on a remote service. Choose a local model.\n"},
		{"no local models", emptyTags, nil, []string{"--provider", "ollama", "Hi"}, 1,
			"neurobrau: error: No local Ollama models are installed (cloud models are not used). Install one with: ollama pull <model>\n"},
		{"server error", tags, respond(500, `{"error":"out of memory"}`), []string{"--provider", "ollama", "Hi"}, 1,
			"neurobrau: error: Ollama API error (HTTP 500): out of memory\n"},
		{"malformed reply", tags, respond(200, `not json`), []string{"--provider", "ollama", "Hi"}, 1,
			"neurobrau: error: Invalid response from Ollama: expected a JSON object.\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chat := tt.chat
			if chat == nil {
				chat = func(w http.ResponseWriter, r *http.Request) { t.Error("chat request must not be sent") }
			}
			startFakeOllama(t, tt.tags, chat)
			code, stdout, stderr := run(t, tt.args...)
			if code != tt.wantCode || stdout != "" || stderr != tt.wantErr {
				t.Errorf("got code=%d stdout=%q stderr=%q; want code=%d stderr=%q", code, stdout, stderr, tt.wantCode, tt.wantErr)
			}
		})
	}
}

func TestOllamaConnectionRefused(t *testing.T) {
	isolateEnv(t)
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	t.Setenv("OLLAMA_HOST", url)

	code, stdout, stderr := run(t, "--provider", "ollama", "Hi")
	want := "neurobrau: error: Could not connect to Ollama at " + url + ". Is the Ollama service running?\n"
	if code != 7 || stdout != "" || stderr != want {
		t.Errorf("got code=%d stdout=%q stderr=%q; want code=7 stderr=%q", code, stdout, stderr, want)
	}
}

func TestInterruptCancelsRequest(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	startFakeOllama(t, tags, func(w http.ResponseWriter, r *http.Request) {
		cancel() // like Ctrl+C while the model is generating
		<-r.Context().Done()
	})
	code, stdout, stderr := runCtx(t, ctx, "--provider", "ollama", "Hi")
	if code != ExitInterrupted || stdout != "" || stderr != "\nneurobrau: interrupted\n" {
		t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

// fakeKey is a synthetic credential; tests never use a real API key.
const fakeKey = "sk-test-synthetic-key-0000"

// fakeOpenAI starts an httptest OpenAI server, points the SDK at it with a
// fake key, and records the Responses API request bodies.
type fakeOpenAI struct {
	mu       sync.Mutex
	requests []map[string]any
}

func startFakeOpenAI(t *testing.T, handler http.HandlerFunc) *fakeOpenAI {
	t.Helper()
	isolateEnv(t)
	f := &fakeOpenAI{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method+" "+r.URL.Path != "POST /v1/responses" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		f.requests = append(f.requests, body)
		f.mu.Unlock()
		handler(w, r)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OPENAI_BASE_URL", srv.URL+"/v1")
	t.Setenv("OPENAI_API_KEY", fakeKey)
	return f
}

func (f *fakeOpenAI) models() []any {
	f.mu.Lock()
	defer f.mu.Unlock()
	var models []any
	for _, r := range f.requests {
		models = append(models, r["model"])
	}
	return models
}

// openAIReply answers with JSON; Retry-After-Ms: 0 keeps SDK retries instant.
func openAIReply(status int, body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After-Ms", "0")
		w.WriteHeader(status)
		io.WriteString(w, body)
	}
}

const openAIAnswer = `{"id":"resp_1","status":"completed","model":"gpt-6-luna","output":[
 {"type":"reasoning","summary":[{"type":"summary_text","text":"SECRET"}]},
 {"type":"message","role":"assistant","content":[{"type":"output_text","text":"A neural network is a model."}]}],
 "usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12}}`

func TestOpenAIOneShot(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantModel string
	}{
		{"openai is the default provider", []string{"What is a neural network?"}, "gpt-6-luna"},
		{"explicit provider", []string{"--provider", "openai", "Hello"}, "gpt-6-luna"},
		{"explicit model", []string{"--provider", "openai", "--model", "gpt-6-luna", "Hello"}, "gpt-6-luna"},
		{"any model ID", []string{"--model", "ft:gpt-4.1:org::abc", "Hello"}, "ft:gpt-4.1:org::abc"},
		{"prompt before flags", []string{"Hello", "--provider=openai"}, "gpt-6-luna"},
		{"prompt after terminator", []string{"--", "--version"}, "gpt-6-luna"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
			code, stdout, stderr := run(t, tt.args...)
			if code != ExitOK || stderr != "" {
				t.Fatalf("got code=%d stderr=%q", code, stderr)
			}
			if want := "A neural network is a model.\n"; stdout != want {
				t.Errorf("stdout = %q, want %q", stdout, want)
			}
			if got := f.models(); len(got) != 1 || got[0] != tt.wantModel {
				t.Errorf("requests for models %v, want [%s]", got, tt.wantModel)
			}
		})
	}
}

func TestOpenAIMissingKey(t *testing.T) {
	f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
	os.Unsetenv("OPENAI_API_KEY")
	for _, args := range [][]string{{"Hi"}, {"--provider", "openai", "--model", "gpt-5", "Hi"}} {
		code, stdout, stderr := run(t, args...)
		want := "neurobrau: error: OPENAI_API_KEY is not set. Export it in your environment and try again.\n"
		if code != 3 || stdout != "" || stderr != want {
			t.Errorf("%v: got code=%d stdout=%q stderr=%q", args, code, stdout, stderr)
		}
	}
	if n := len(f.models()); n != 0 {
		t.Errorf("%d requests without a key, want 0", n)
	}
}

func TestOpenAIErrors(t *testing.T) {
	tests := []struct {
		name     string
		handler  http.HandlerFunc
		wantCode int
		wantErr  string
	}{
		{"authentication", openAIReply(401, `{"error":{"message":"Incorrect API key provided: `+fakeKey+`"}}`), 4,
			"neurobrau: error: OpenAI authentication failed. Check that OPENAI_API_KEY is valid.\n"},
		{"permission", openAIReply(403, `{"error":{"message":"no access"}}`), 4,
			"neurobrau: error: OpenAI denied access (HTTP 403). Check that OPENAI_API_KEY is valid and allowed to use this model.\n"},
		{"rate limit", openAIReply(429, `{"error":{"message":"quota"}}`), 5,
			"neurobrau: error: OpenAI rate limit or quota exceeded. Wait and try again, or check your plan and billing.\n"},
		{"bad request", openAIReply(400, `{"error":{"message":"Invalid value for 'input'."}}`), 1,
			"neurobrau: error: OpenAI API error (HTTP 400): Invalid value for 'input'.\n"},
		{"server error", openAIReply(500, `{"error":{"message":"The server had an error."}}`), 1,
			"neurobrau: error: OpenAI API error (HTTP 500): The server had an error.\n"},
		{"failed response", openAIReply(200, `{"status":"failed","error":{"message":"The model crashed."},"output":[]}`), 1,
			"neurobrau: error: OpenAI response failed: The model crashed.\n"},
		{"refusal", openAIReply(200, `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I can't help."}]}]}`), 1,
			"neurobrau: error: OpenAI refused the request: I can't help.\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startFakeOpenAI(t, tt.handler)
			code, stdout, stderr := run(t, "Hi")
			if code != tt.wantCode || stdout != "" || stderr != tt.wantErr {
				t.Errorf("got code=%d stdout=%q stderr=%q; want code=%d stderr=%q", code, stdout, stderr, tt.wantCode, tt.wantErr)
			}
			if strings.Contains(stderr, fakeKey) {
				t.Error("stderr exposes the API key")
			}
		})
	}
}

// A non-streaming answer that stopped early is printed as is with exit 0,
// like the Python reference; exit 8 is reserved for streamed answers.
func TestOpenAIIncompleteResponse(t *testing.T) {
	tests := []struct {
		name, body, wantStdout string
	}{
		{"partial text", `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Partial ans"}]}]}`, "Partial ans\n"},
		{"no text", `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}`, "\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startFakeOpenAI(t, openAIReply(200, tt.body))
			code, stdout, stderr := run(t, "Hi")
			if code != ExitOK || stdout != tt.wantStdout || stderr != "" {
				t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
			}
		})
	}
}

func TestOpenAIConnectionRefused(t *testing.T) {
	isolateEnv(t)
	t.Setenv("OPENAI_API_KEY", fakeKey) // OPENAI_BASE_URL points at a closed port
	code, stdout, stderr := run(t, "Hi")
	want := "neurobrau: error: Could not connect to OpenAI. Check your network connection.\n"
	if code != 7 || stdout != "" || stderr != want {
		t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

func TestOpenAIInterrupt(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	startFakeOpenAI(t, func(w http.ResponseWriter, r *http.Request) {
		cancel() // like Ctrl+C while the model is generating
		<-r.Context().Done()
	})
	code, stdout, stderr := runCtx(t, ctx, "Hi")
	if code != ExitInterrupted || stdout != "" || stderr != "\nneurobrau: interrupted\n" {
		t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

func TestWithoutOpenAIKey(t *testing.T) {
	startFakeOllama(t, tags, respond(200, answer)) // isolateEnv: no OpenAI key
	for _, args := range [][]string{{"--help"}, {"--version"}, {"--provider", "ollama", "Hi"}} {
		if code, _, stderr := run(t, args...); code != ExitOK || stderr != "" {
			t.Errorf("%v: code=%d stderr=%q", args, code, stderr)
		}
	}
}

func (f *fakeOllama) bodies() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.chats)
}

func (f *fakeOpenAI) bodies() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.requests)
}

// --- Task 002C: generation settings ---

func effort(e string) any { return map[string]any{"effort": e} }

func TestOpenAIGenerationSettings(t *testing.T) {
	brief := llm.StyleInstructions(llm.StyleBrief)
	detailed := llm.StyleInstructions(llm.StyleDetailed)
	tests := []struct {
		name string
		args []string
		want map[string]any // expected fields; nil value: field must be absent
	}{
		// Regression: gpt-6-luna gets NeuroBräu's default effort "none".
		{"default reasoning none", []string{"Hi"},
			map[string]any{"model": "gpt-6-luna", "reasoning": effort("none"), "store": false, "instructions": nil, "max_output_tokens": nil}},
		{"model default omits reasoning", []string{"--reasoning-effort", "model-default", "Hi"},
			map[string]any{"reasoning": nil, "store": false}},
		{"explicit high", []string{"--reasoning-effort", "high", "Hi"},
			map[string]any{"reasoning": effort("high")}},
		{"flags after prompt", []string{"Hi", "--reasoning-effort", "xhigh", "--max-output-tokens", "300"},
			map[string]any{"reasoning": effort("xhigh"), "max_output_tokens": float64(300)}},
		{"unknown model, no flag", []string{"--model", "my-new-model", "Hi"},
			map[string]any{"reasoning": nil}},
		{"unknown model, explicit effort", []string{"--model", "my-new-model", "--reasoning-effort", "max", "Hi"},
			map[string]any{"reasoning": effort("max")}},
		{"known model without none", []string{"--model", "gpt-5", "Hi"},
			map[string]any{"reasoning": nil}},
		{"none on a non-reasoning model", []string{"--model", "gpt-4o", "--reasoning-effort", "none", "Hi"},
			map[string]any{"reasoning": nil}},
		{"max output tokens", []string{"--max-output-tokens", "256", "Hi"},
			map[string]any{"max_output_tokens": float64(256)}},
		{"at the known limit", []string{"--model", "gpt-4o", "--max-output-tokens", "16384", "Hi"},
			map[string]any{"max_output_tokens": float64(16384)}},
		{"normal style", []string{"--answer-style", "normal", "Hi"}, map[string]any{"instructions": nil}},
		{"brief style", []string{"--answer-style", "brief", "Hi"}, map[string]any{"instructions": brief}},
		{"detailed style", []string{"--answer-style", "detailed", "Hi"}, map[string]any{"instructions": detailed}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
			code, stdout, stderr := run(t, tt.args...)
			if code != ExitOK || stdout != "A neural network is a model.\n" || stderr != "" {
				t.Fatalf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
			}
			body := f.bodies()[0]
			for key, want := range tt.want {
				got, present := body[key]
				if want == nil && present || want != nil && !reflect.DeepEqual(got, want) {
					t.Errorf("%s = %v (present %v), want %v", key, got, present, want)
				}
			}
			if input := body["input"].([]any); len(input) != 1 {
				t.Errorf("input = %v, want only the user message", input)
			}
		})
	}
}

func TestInvalidGenerationSettings(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"unsupported effort for known model", []string{"--model", "gpt-5", "--reasoning-effort", "none", "Hi"},
			"neurobrau: error: gpt-5 supports reasoning: Minimal, Low, Medium, High; 'none' is not available.\n"},
		{"effort on non-reasoning model", []string{"--model", "gpt-4o", "--reasoning-effort", "high", "Hi"},
			"neurobrau: error: gpt-4o does not support reasoning; 'high' is not available.\n"},
		{"snapshot uses base model table", []string{"--model", "gpt-5-2025-08-07", "--reasoning-effort", "xhigh", "Hi"},
			"neurobrau: error: gpt-5-2025-08-07 supports reasoning: Minimal, Low, Medium, High; 'xhigh' is not available.\n"},
		{"over known limit", []string{"--model", "gpt-4o", "--max-output-tokens", "16385", "Hi"},
			"neurobrau: error: 16385 exceeds the limit of 16384 output tokens for gpt-4o.\n"},
		{"invalid effort name", []string{"--reasoning-effort", "turbo", "Hi"},
			`neurobrau: error: argument --reasoning-effort: invalid choice: "turbo" (choose from none, minimal, low, medium, high, xhigh, max, model-default)` + "\n"},
		{"effort is case-sensitive", []string{"--reasoning-effort", "HIGH", "Hi"}, "invalid choice"},
		{"invalid style", []string{"--answer-style", "short", "Hi"},
			`neurobrau: error: argument --answer-style: invalid choice: "short" (choose from brief, normal, detailed)` + "\n"},
		{"zero tokens", []string{"--max-output-tokens", "0", "Hi"},
			"neurobrau: error: argument --max-output-tokens: must be greater than 0, got 0\n"},
		{"negative tokens", []string{"--max-output-tokens", "-5", "Hi"},
			"neurobrau: error: argument --max-output-tokens: must be greater than 0, got -5\n"},
		{"float tokens", []string{"--max-output-tokens", "1.5", "Hi"},
			`neurobrau: error: argument --max-output-tokens: invalid integer: "1.5"` + "\n"},
		{"text tokens", []string{"--max-output-tokens=abc", "Hi"}, `invalid integer: "abc"`},
		{"empty tokens", []string{"--max-output-tokens=", "Hi"}, `invalid integer: ""`},
		{"overflowing tokens", []string{"--max-output-tokens", "99999999999999999999", "Hi"}, "invalid integer"},
		{"missing token value", []string{"Hi", "--max-output-tokens"}, "flag needs an argument: -max-output-tokens"},
		{"stream is not implemented", []string{"--stream", "Hi"}, "flag provided but not defined: -stream"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
			code, stdout, stderr := run(t, tt.args...)
			if code != ExitUsage || stdout != "" || !strings.Contains(stderr, tt.wantErr) {
				t.Errorf("got code=%d stdout=%q stderr=%q; want exit 2 with %q", code, stdout, stderr, tt.wantErr)
			}
			if n := len(f.bodies()); n != 0 {
				t.Errorf("%d chat requests, want 0", n)
			}
		})
	}
}

func TestInvalidSettingsNeedNoKey(t *testing.T) {
	isolateEnv(t) // no OPENAI_API_KEY: the model table is checked first
	code, _, stderr := run(t, "--model", "gpt-4o", "--reasoning-effort", "high", "Hi")
	if code != ExitUsage || !strings.Contains(stderr, "does not support reasoning") {
		t.Errorf("got code=%d stderr=%q", code, stderr)
	}
}

const reasoningRejected = `{"error":{"message":"Unsupported parameter: 'reasoning.effort' is not supported with this model. Key ` + fakeKey + `","type":"invalid_request_error","param":"reasoning.effort"}}`

// rejectReasoning answers 400 when the request has a reasoning field.
func rejectReasoning(f **fakeOpenAI) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		bodies := (*f).bodies()
		if _, ok := bodies[len(bodies)-1]["reasoning"]; ok {
			openAIReply(400, reasoningRejected)(w, r)
			return
		}
		openAIReply(200, openAIAnswer)(w, r)
	}
}

func TestOpenAIRejectedReasoning(t *testing.T) {
	t.Run("built-in default falls back once", func(t *testing.T) {
		var f *fakeOpenAI
		f = startFakeOpenAI(t, rejectReasoning(&f))
		code, stdout, stderr := run(t, "Hi")
		if code != ExitOK || stdout != "A neural network is a model.\n" || stderr != "" {
			t.Fatalf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
		}
		bodies := f.bodies()
		if len(bodies) != 2 || !reflect.DeepEqual(bodies[0]["reasoning"], effort("none")) || bodies[1]["reasoning"] != nil {
			t.Errorf("requests = %v, want reasoning none, then none sent", bodies)
		}
		for _, b := range bodies {
			if b["store"] != false {
				t.Errorf("store = %v, want false", b["store"])
			}
		}
	})
	t.Run("explicit effort is reported", func(t *testing.T) {
		var f *fakeOpenAI
		f = startFakeOpenAI(t, rejectReasoning(&f))
		code, stdout, stderr := run(t, "--reasoning-effort", "low", "private prompt")
		want := "neurobrau: error: gpt-6-luna does not accept reasoning effort 'low' (OpenAI API error (HTTP 400): " +
			"Unsupported parameter: 'reasoning.effort' is not supported with this model. Key [redacted]). " +
			"Choose another value with --reasoning-effort.\n"
		if code != 1 || stdout != "" || stderr != want {
			t.Errorf("got code=%d stdout=%q stderr=%q; want %q", code, stdout, stderr, want)
		}
		if n := len(f.bodies()); n != 1 {
			t.Errorf("%d requests, want 1 (no fallback)", n)
		}
	})
}

func TestUsageOutput(t *testing.T) {
	tests := []struct {
		name, body, wantUsage string
	}{
		{"reported", openAIAnswer, "Token usage: input=5, output=7, total=12\n"},
		{"not reported", `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"A neural network is a model."}]}]}`,
			"Token usage: not reported\n"},
		{"partial, no invented total", `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"A neural network is a model."}]}],"usage":{"output_tokens":3,"output_tokens_details":{"reasoning_tokens":2}}}`,
			"Token usage: output=3\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startFakeOpenAI(t, openAIReply(200, tt.body))
			code, stdout, stderr := run(t, "Hi", "--usage")
			if code != ExitOK || stdout != "A neural network is a model.\n" || stderr != tt.wantUsage {
				t.Errorf("got code=%d stdout=%q stderr=%q; want stderr %q", code, stdout, stderr, tt.wantUsage)
			}
		})
	}
	t.Run("without --usage", func(t *testing.T) {
		startFakeOpenAI(t, openAIReply(200, openAIAnswer))
		if _, _, stderr := run(t, "Hi"); stderr != "" {
			t.Errorf("stderr = %q, want empty", stderr)
		}
	})
	t.Run("ollama", func(t *testing.T) {
		startFakeOllama(t, tags, respond(200, answer))
		code, stdout, stderr := run(t, "--provider", "ollama", "--usage", "Hi")
		if code != ExitOK || stdout != "Neural networks learn.\n" || stderr != "Token usage: input=10, output=4, total=14\n" {
			t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
		}
	})
}

func TestCompletedEmptyVersusFailed(t *testing.T) {
	t.Run("completed with empty text", func(t *testing.T) {
		startFakeOpenAI(t, openAIReply(200, `{"status":"completed","output":[]}`))
		if code, stdout, stderr := run(t, "Hi"); code != ExitOK || stdout != "\n" || stderr != "" {
			t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
		}
	})
	t.Run("failed", func(t *testing.T) {
		startFakeOpenAI(t, openAIReply(200, `{"status":"failed","error":{"message":"boom"},"output":[]}`))
		if code, stdout, stderr := run(t, "Hi"); code != 1 || stdout != "" || stderr != "neurobrau: error: OpenAI response failed: boom\n" {
			t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
		}
	})
}

// Ollama models for the CLI: thinking on/off (context 4096), named levels,
// and no thinking.
const (
	thinkingTags = `{"models":[{"model":"qwen3.5:4b","capabilities":["completion","thinking"],"details":{"context_length":4096}},` +
		`{"model":"gpt-oss:20b","capabilities":["completion","thinking"]},{"model":"llama3.2:3b","capabilities":["completion"]}]}`
	lengthAnswer = `{"message":{"role":"assistant","content":"Cut off mid"},"done":true,"done_reason":"length","prompt_eval_count":3,"eval_count":2}`
)

var thinkingShows = map[string]string{
	"qwen3.5:4b":  `{"capabilities":["completion","thinking"]}`,
	"gpt-oss:20b": `{"capabilities":["completion","thinking"],"thinking":{"values":["low","medium","high"]}}`,
	"llama3.2:3b": `{"capabilities":["completion"]}`,
}

func TestOllamaGenerationSettings(t *testing.T) {
	detailed := llm.StyleInstructions(llm.StyleDetailed)
	tests := []struct {
		name string
		args []string
		want map[string]any // nil value: field must be absent
	}{
		{"default thinking off", []string{"--model", "qwen3.5:4b", "Hi"}, map[string]any{"think": false, "options": nil}},
		{"none means off", []string{"--model", "qwen3.5:4b", "--reasoning-effort", "none", "Hi"}, map[string]any{"think": false}},
		{"effort means on", []string{"--model", "qwen3.5:4b", "--reasoning-effort", "low", "Hi"}, map[string]any{"think": true}},
		{"named level", []string{"--model", "gpt-oss:20b", "--reasoning-effort", "high", "Hi"}, map[string]any{"think": "high"}},
		{"level model default", []string{"--model", "gpt-oss:20b", "--reasoning-effort", "model-default", "Hi"}, map[string]any{"think": nil}},
		{"no thinking", []string{"--model", "llama3.2:3b", "--reasoning-effort", "none", "Hi"}, map[string]any{"think": nil}},
		{"num_predict", []string{"--model", "llama3.2:3b", "--max-output-tokens", "256", "Hi"},
			map[string]any{"options": map[string]any{"num_predict": float64(256)}}},
		{"at context length", []string{"--model", "qwen3.5:4b", "--max-output-tokens", "4096", "Hi"},
			map[string]any{"options": map[string]any{"num_predict": float64(4096)}}},
		{"detailed style", []string{"--model", "llama3.2:3b", "--answer-style", "detailed", "Hi"},
			map[string]any{"messages": []any{
				map[string]any{"role": "system", "content": detailed},
				map[string]any{"role": "user", "content": "Hi"},
			}}},
		{"normal style", []string{"--model", "llama3.2:3b", "--answer-style", "normal", "Hi"},
			map[string]any{"messages": []any{map[string]any{"role": "user", "content": "Hi"}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := startFakeOllamaWithShow(t, thinkingTags, thinkingShows, respond(200, answer))
			code, stdout, stderr := run(t, append([]string{"--provider", "ollama"}, tt.args...)...)
			if code != ExitOK || stdout != "Neural networks learn.\n" || stderr != "" {
				t.Fatalf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
			}
			body := f.bodies()[0]
			for key, want := range tt.want {
				got, present := body[key]
				if want == nil && present || want != nil && !reflect.DeepEqual(got, want) {
					t.Errorf("%s = %v (present %v), want %v", key, got, present, want)
				}
			}
		})
	}
}

func TestOllamaInvalidSettings(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"no model default on on/off model", []string{"--model", "qwen3.5:4b", "--reasoning-effort", "model-default", "Hi"},
			"neurobrau: error: qwen3.5:4b has no model-default reasoning choice (qwen3.5:4b supports reasoning: Off, On).\n"},
		{"thinking on a model without it", []string{"--model", "llama3.2:3b", "--reasoning-effort", "high", "Hi"},
			"neurobrau: error: llama3.2:3b does not support reasoning; 'high' is not available.\n"},
		{"level model without off", []string{"--model", "gpt-oss:20b", "--reasoning-effort", "none", "Hi"},
			"neurobrau: error: gpt-oss:20b supports reasoning: Low, Medium, High; 'none' is not available.\n"},
		{"over context length", []string{"--model", "qwen3.5:4b", "--max-output-tokens", "4097", "Hi"},
			"neurobrau: error: 4097 exceeds the limit of 4096 output tokens for qwen3.5:4b.\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := startFakeOllamaWithShow(t, thinkingTags, thinkingShows, respond(200, answer))
			code, stdout, stderr := run(t, append([]string{"--provider", "ollama"}, tt.args...)...)
			if code != ExitUsage || stdout != "" || stderr != tt.wantErr {
				t.Errorf("got code=%d stdout=%q stderr=%q; want %q", code, stdout, stderr, tt.wantErr)
			}
			if n := len(f.bodies()); n != 0 {
				t.Errorf("%d chat requests, want 0", n)
			}
		})
	}
}

// A non-streaming answer cut off at the token limit is printed as is with
// exit 0, like the Python reference.
func TestOllamaLengthLimitedAnswer(t *testing.T) {
	startFakeOllama(t, tags, respond(200, lengthAnswer))
	code, stdout, stderr := run(t, "--provider", "ollama", "--max-output-tokens", "2", "--usage", "Hi")
	if code != ExitOK || stdout != "Cut off mid\n" || stderr != "Token usage: input=3, output=2, total=5\n" {
		t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

// Help and version never contact a server, with or without credentials.
func TestHelpAndVersionMakeNoRequests(t *testing.T) {
	isolateEnv(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
	}))
	defer srv.Close()
	t.Setenv("OPENAI_BASE_URL", srv.URL+"/v1")
	t.Setenv("OLLAMA_HOST", srv.URL)
	for _, args := range [][]string{{"--help"}, {"-h"}, {"--version"}, {"--plain", "--help"}, {"--usage", "--help"}, {"--reasoning-effort", "high", "--version"}} {
		if code, _, stderr := run(t, args...); code != ExitOK || stderr != "" {
			t.Errorf("%v: code=%d stderr=%q", args, code, stderr)
		}
	}
}
