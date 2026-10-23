package cli

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"syscall"
	"testing"
)

func ollamaLine(text string) string {
	return `{"model":"llama3.2:3b","message":{"role":"assistant","content":` + quoteJSON(text) + `},"done":false}` + "\n"
}

func quoteJSON(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)
	return `"` + r.Replace(s) + `"`
}

const (
	ollamaDone = `{"model":"llama3.2:3b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",` +
		`"prompt_eval_count":12,"eval_count":5,"total_duration":2500000000,"load_duration":1000000000,"eval_duration":1200000000}` + "\n"
	ollamaLength = `{"model":"llama3.2:3b","message":{"role":"assistant","content":""},"done":true,"done_reason":"length","prompt_eval_count":3,"eval_count":7}` + "\n"
)

// chunks writes the pieces one by one, flushing after each.
func chunks(pieces ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		for _, p := range pieces {
			io.WriteString(w, p)
			w.(http.Flusher).Flush()
		}
	}
}

func sseEvent(eventType, data string) string {
	return "event: " + eventType + "\ndata: " + data + "\n\n"
}

func openAIDelta(text string) string {
	return sseEvent("response.output_text.delta", `{"type":"response.output_text.delta","delta":`+quoteJSON(text)+`}`)
}

const openAICompleted = `{"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-6-luna",` +
	`"output":[],"usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12}}}`

func openAIStream(events ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunks(events...)(w, r)
	}
}

func TestStreamOllama(t *testing.T) {
	f := startFakeOllama(t, tags, chunks(
		`{"message":{"content":"","thinking":"SECRET THOUGHT"},"done":false}`+"\n",
		ollamaLine("<think>SECRET"), ollamaLine(" PLAN</think>Neural "), ollamaLine("networks 🧠\nlearn."), ollamaDone))
	for _, args := range [][]string{
		{"--provider", "ollama", "--model", "llama3.2:3b", "--stream", "--usage", "Hi"},
		{"--provider", "ollama", "--model", "llama3.2:3b", "Hi", "--usage", "--stream"}, // flags after the prompt
	} {
		code, stdout, stderr := run(t, args...)
		if code != ExitOK || stdout != "Neural networks 🧠\nlearn.\n" ||
			stderr != "Token usage: input=12, output=5, total=17\nDuration: 2.50 s (load 1.00 s, generation 1.20 s)\n" {
			t.Errorf("%v: code=%d stdout=%q stderr=%q", args, code, stdout, stderr)
		}
	}
	if body := f.chats[0]; body["stream"] != true || body["model"] != "llama3.2:3b" {
		t.Errorf("request %v", body)
	}
	if _, err := os.Stat(filepath.Dir(statePaths(t).Conversations)); err == nil {
		t.Error("a one-shot stream created the history directory")
	}
}

func TestStreamOpenAI(t *testing.T) {
	f := startFakeOpenAI(t, openAIStream(openAIDelta("A neural "), openAIDelta("network."), sseEvent("response.completed", openAICompleted)))
	code, stdout, stderr := run(t, "--stream", "--usage", "--answer-style", "brief", "Hi")
	if code != ExitOK || stdout != "A neural network.\n" || stderr != "Token usage: input=5, output=7, total=12\n" {
		t.Errorf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
	body := f.bodies()[0]
	if body["stream"] != true || body["store"] != false || !reflect.DeepEqual(body["reasoning"], effort("none")) ||
		body["instructions"] != llmStyle("brief") {
		t.Errorf("request %v", body)
	}
}

func TestStreamOutcomes(t *testing.T) {
	tests := []struct {
		name           string
		handler        http.HandlerFunc
		args           []string
		code           int
		stdout, stderr string
	}{
		{"empty completed answer", chunks(ollamaDone), nil, ExitOK, "\n", ""},
		{"incomplete", chunks(ollamaLine("Part"), ollamaLength), []string{"--usage"}, ExitIncomplete, "Part\n",
			"neurobrau: warning: response incomplete: the max output token limit was reached.\nToken usage: input=3, output=7, total=10\n"},
		{"incomplete without text", chunks(ollamaLength), nil, ExitIncomplete, "",
			"neurobrau: warning: response incomplete: the max output token limit was reached.\n"},
		{"error line", chunks(ollamaLine("Part"), `{"error":"model crashed"}`+"\n"), nil, 1, "Part\n", "neurobrau: error: Ollama error: model crashed\n"},
		{"end without done", chunks(ollamaLine("Part")), nil, 7, "Part\n",
			"neurobrau: error: Ollama stream ended before the response was complete.\n"},
		{"HTTP 404", respond(404, `{"error":"not found"}`), nil, 1, "",
			"neurobrau: error: Ollama model 'llama3.2:3b' was not found: not found. Install it yourself with: ollama pull llama3.2:3b\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startFakeOllama(t, tags, tt.handler)
			code, stdout, stderr := run(t, append([]string{"--provider", "ollama", "--model", "llama3.2:3b", "--stream", "Hi"}, tt.args...)...)
			if code != tt.code || stdout != tt.stdout || stderr != tt.stderr {
				t.Errorf("code=%d stdout=%q stderr=%q\nwant %d %q %q", code, stdout, stderr, tt.code, tt.stdout, tt.stderr)
			}
		})
	}
}

func TestStreamOpenAIErrorCodes(t *testing.T) {
	tests := []struct {
		name    string
		handler http.HandlerFunc
		code    int
	}{
		{"401", openAIReply(401, `{"error":{"message":"bad key"}}`), 4},
		{"403", openAIReply(403, `{"error":{"message":"denied"}}`), 4},
		{"429", openAIReply(429, `{"error":{"message":"slow down"}}`), 5},
		{"stream error", openAIStream(openAIDelta("Part"), sseEvent("error", `{"type":"error","message":"busy"}`)), 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startFakeOpenAI(t, tt.handler)
			code, _, stderr := run(t, "--stream", "Hi")
			if code != tt.code || !strings.HasPrefix(stderr, "neurobrau: error: ") || strings.Contains(stderr, fakeKey) {
				t.Errorf("code=%d stderr=%q", code, stderr)
			}
		})
	}
	t.Run("missing key", func(t *testing.T) {
		isolateEnv(t)
		if code, stdout, _ := run(t, "--stream", "Hi"); code != 3 || stdout != "" {
			t.Errorf("code=%d", code)
		}
	})
	t.Run("connection refused", func(t *testing.T) {
		isolateEnv(t)
		t.Setenv("OPENAI_API_KEY", fakeKey)
		if code, _, _ := run(t, "--stream", "Hi"); code != 7 {
			t.Errorf("code=%d", code)
		}
	})
}

func TestStreamWithoutPrompt(t *testing.T) {
	startFakeOpenAI(t, openAIReply(200, openAIAnswer))
	code, stdout, stderr := chatRun(t, "Hi\n", "--stream")
	if code != ExitUsage || stdout != "" || !strings.Contains(stderr, "--stream needs a prompt") {
		t.Errorf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

// signalWriter records writes and signals the first one.
type signalWriter struct {
	mu    sync.Mutex
	buf   bytes.Buffer
	first chan struct{}
	once  sync.Once
}

func (w *signalWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.once.Do(func() { close(w.first) })
	return w.buf.Write(p)
}

func (w *signalWriter) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.String()
}

// TestStreamIsIncremental shows that text reaches stdout before the
// answer is finished: the server holds the rest back until it was written.
func TestStreamIsIncremental(t *testing.T) {
	out := &signalWriter{first: make(chan struct{})}
	startFakeOllama(t, tags, func(w http.ResponseWriter, r *http.Request) {
		chunks(ollamaLine("First "))(w, r)
		<-out.first // stdout got the first piece while the request runs
		chunks(ollamaLine("second."), ollamaDone)(w, r)
	})
	var errOut bytes.Buffer
	code := Run(context.Background(), []string{"--provider", "ollama", "--model", "llama3.2:3b", "--stream", "Hi"},
		Streams{Stdout: out, Stderr: &errOut})
	if code != ExitOK || out.String() != "First second.\n" || errOut.String() != "" {
		t.Errorf("code=%d stdout=%q stderr=%q", code, out.String(), errOut.String())
	}
}

func TestStreamInterrupt(t *testing.T) {
	interrupts := make(chan os.Signal, 1)
	closed := make(chan struct{})
	out := &signalWriter{first: make(chan struct{})}
	startFakeOllama(t, tags, func(w http.ResponseWriter, r *http.Request) {
		chunks(ollamaLine("Part"))(w, r)
		<-out.first
		interrupts <- os.Interrupt // Ctrl+C while the server stalls
		<-r.Context().Done()
		close(closed)
	})
	var errOut bytes.Buffer
	code := Run(context.Background(), []string{"--provider", "ollama", "--model", "llama3.2:3b", "--stream", "Hi"},
		Streams{Stdout: out, Stderr: &errOut, Interrupts: interrupts})
	if code != ExitInterrupted || out.String() != "Part\n" || errOut.String() != "neurobrau: interrupted\n" {
		t.Errorf("code=%d stdout=%q stderr=%q", code, out.String(), errOut.String())
	}
	<-closed // the connection was closed
}

// pipeWriter fails like a pipe whose reader went away.
type pipeWriter struct{ writes int }

func (w *pipeWriter) Write(p []byte) (int, error) {
	w.writes++
	return 0, syscall.EPIPE
}

func TestStreamBrokenPipe(t *testing.T) {
	closed := make(chan struct{})
	startFakeOllama(t, tags, func(w http.ResponseWriter, r *http.Request) {
		for r.Context().Err() == nil { // generates until the client goes away
			chunks(ollamaLine("more "))(w, r)
		}
		close(closed)
	})
	out := &pipeWriter{}
	var errOut bytes.Buffer
	code := Run(context.Background(), []string{"--provider", "ollama", "--model", "llama3.2:3b", "--stream", "Hi"},
		Streams{Stdout: out, Stderr: &errOut})
	if code != ExitBrokenPipe || errOut.String() != "" || out.writes != 1 {
		t.Errorf("code=%d stderr=%q writes=%d", code, errOut.String(), out.writes)
	}
	<-closed
}
