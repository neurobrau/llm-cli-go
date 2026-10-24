package cli

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// statePaths are the isolated locations isolateEnv set up.
func statePaths(t *testing.T) storage.Paths {
	t.Helper()
	paths, err := storage.ResolvePaths(os.Getenv, os.Getenv("HOME"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(paths.Conversations, os.TempDir()) || !strings.HasPrefix(paths.Settings, os.TempDir()) {
		t.Fatalf("paths %+v are not temporary", paths)
	}
	return paths
}

// chatRun runs the CLI with input on stdin.
func chatRun(t *testing.T, input string, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errOut bytes.Buffer
	code = Run(context.Background(), args, Streams{Stdin: strings.NewReader(input), Stdout: &out, Stderr: &errOut})
	return code, out.String(), errOut.String()
}

func writeTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func fixture(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "storage", "testdata", "python", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// tree returns every file under dir with its content and mode.
func tree(t *testing.T, dir string) map[string]string {
	t.Helper()
	files := map[string]string{}
	filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			files[path] = "error: " + err.Error()
			return nil
		}
		info, _ := d.Info()
		content := ""
		if !d.IsDir() {
			data, _ := os.ReadFile(path)
			content = string(data)
		}
		files[path] = info.Mode().String() + " " + content
		return nil
	})
	return files
}

// messages returns role/content pairs of a recorded request body (OpenAI
// "input", Ollama "messages"), without system messages.
func messages(body map[string]any) []string {
	items, _ := body["input"].([]any)
	if items == nil {
		items, _ = body["messages"].([]any)
	}
	var out []string
	for _, item := range items {
		m := item.(map[string]any)
		if m["role"] == "system" {
			continue
		}
		out = append(out, m["role"].(string)+": "+m["content"].(string))
	}
	return out
}

const chatBanner = "NeuroBräu Go CLI 0.4.0-dev\nProvider: openai | Model: gpt-6-luna\nType /help for commands, /exit to quit.\n\n"

func TestInteractiveStartsWithoutPrompt(t *testing.T) {
	for _, args := range [][]string{{}, {"--plain"}, {"--plain", "--no-history"}} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
			code, stdout, stderr := chatRun(t, "", args...)
			if code != ExitOK || stdout != chatBanner+"YOU > \nGoodbye.\n" || stderr != "" {
				t.Errorf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
			}
			if len(f.models()) != 0 {
				t.Error("requests were sent")
			}
			if _, err := os.Stat(filepath.Dir(statePaths(t).Conversations)); err == nil {
				t.Error("an empty chat created the history directory")
			}
		})
	}
}

func TestOneShotWritesNoHistory(t *testing.T) {
	startFakeOpenAI(t, openAIReply(200, openAIAnswer))
	paths := statePaths(t)
	writeTestFile(t, paths.Settings, `{"schema_version": 1, "answer_style": "brief"}`)
	if code, stdout, _ := run(t, "--plain", "--no-history", "Hi"); code != ExitOK || stdout != "A neural network is a model.\n" {
		t.Fatalf("code=%d stdout=%q", code, stdout)
	}
	if code, _, _ := run(t, "Hi"); code != ExitOK {
		t.Fatal(code)
	}
	if _, err := os.Stat(filepath.Dir(paths.Conversations)); err == nil {
		t.Error("one-shot created the history directory")
	}
}

func TestOneShotIgnoresSavedSettings(t *testing.T) {
	f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
	writeTestFile(t, statePaths(t).Settings, `{"schema_version": 1, "answer_style": "brief", "providers": {"openai":
	  {"models": {"gpt-6-luna": {"reasoning": "high", "max_output_tokens": 99}}}}}`)
	run(t, "Hi")
	body := f.bodies()[0]
	if body["instructions"] != nil || body["max_output_tokens"] != nil || !reflect.DeepEqual(body["reasoning"], effort("none")) {
		t.Errorf("one-shot request %v", body)
	}
}

// The reference example of docs/tui-spec.md, section 15, through the real
// provider: the reported tier and cache details give an exact estimate.
func TestInteractiveEstimatedCost(t *testing.T) {
	startFakeOpenAI(t, openAIReply(200, `{"id":"resp_1","status":"completed","model":"gpt-6-luna",
	 "service_tier":"default","output":[{"type":"message","role":"assistant",
	 "content":[{"type":"output_text","text":"A neural network is a model."}]}],
	 "usage":{"input_tokens":2500,"output_tokens":300,"total_tokens":2800,
	  "input_tokens_details":{"cached_tokens":1600,"cache_write_tokens":100},
	  "output_tokens_details":{"reasoning_tokens":50}}}`))
	code, stdout, stderr := chatRun(t, "Hi\n/usage\n/exit\n")
	want := `YOU > Conversation usage (1 request(s), as reported by the API):
Input tokens: 2,500
  of which cached: 1,600
  of which cache writes: 100
Output tokens: 300
  of which reasoning: 50
Total tokens: 2,800
Latest request input: 2,500 tokens (the whole history is sent, and billed, with every request)
Estimated cost: $0.000259 (ESTIMATED, not a billed amount; pricing 2026-10-22)
YOU > Goodbye.
`
	if code != ExitOK || !strings.HasSuffix(stdout, want) || stderr != "" {
		t.Errorf("code=%d stderr=%q stdout:\n%s", code, stderr, stdout)
	}
	// Costs are derived when shown, never stored.
	files, _ := filepath.Glob(filepath.Join(statePaths(t).Conversations, "*.json"))
	if len(files) != 1 {
		t.Fatalf("saved %v", files)
	}
	data, _ := os.ReadFile(files[0])
	if !strings.Contains(string(data), `"cache_write_tokens": 100`) || strings.Contains(string(data), "cost") {
		t.Errorf("conversation file:\n%s", data)
	}
}

func TestInteractiveOpenAIConversation(t *testing.T) {
	f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
	paths := statePaths(t)
	code, stdout, stderr := chatRun(t, "Hi\nAnd again?\n/usage\n/exit\n", "--usage")
	if code != ExitOK {
		t.Fatalf("code=%d stderr=%q", code, stderr)
	}
	if !strings.HasPrefix(stdout, chatBanner+"YOU > AI > A neural network is a model.\nYOU > AI > A neural network is a model.\nYOU > Conversation usage (2 request(s)") {
		t.Errorf("stdout %q", stdout)
	}
	if !strings.Contains(stdout, "Input tokens: 10\nOutput tokens: 14\nTotal tokens: 24\nLatest request input: 5 tokens") ||
		!strings.HasSuffix(stdout, "Estimated cost: approximately $0.000008 (ESTIMATED, not a billed amount; pricing 2026-10-22; "+
			"Standard tier assumed (tier not reported); cached tokens not reported (all input priced as uncached); "+
			"cache writes not reported (priced as ordinary input))\nYOU > Goodbye.\n") {
		t.Errorf("usage report %q", stdout)
	}
	if stderr != strings.Repeat("Token usage: input=5, output=7, total=12\n", 2) {
		t.Errorf("stderr %q", stderr)
	}
	bodies := f.bodies()
	want := []string{"user: Hi", "assistant: A neural network is a model.", "user: And again?"}
	if len(bodies) != 2 || !reflect.DeepEqual(messages(bodies[1]), want) || bodies[1]["store"] != false ||
		!reflect.DeepEqual(bodies[1]["reasoning"], effort("none")) {
		t.Errorf("second request %v", bodies[len(bodies)-1])
	}

	files, _ := filepath.Glob(filepath.Join(paths.Conversations, "*.json"))
	if len(files) != 1 {
		t.Fatalf("files %v", files)
	}
	c, err := storage.NewHistoryStore(paths.Conversations).Load(files[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Messages) != 4 || len(c.Requests) != 2 || c.Provider != "openai" || c.Model != "gpt-6-luna" {
		t.Errorf("saved %+v", c)
	}
	if data, _ := os.ReadFile(paths.Active); string(data) != `{"schema_version": 1, "provider": "openai", "model": "gpt-6-luna"}`+"\n" {
		t.Errorf("active.json %q", data)
	}
	if _, err := os.Stat(paths.Settings); err == nil {
		t.Error("settings.json written")
	}

	// The next start continues the conversation.
	code, stdout, _ = chatRun(t, "Third\n")
	if code != ExitOK || !strings.Contains(stdout, "(4 messages). Type /reset to start a new one.\n\n") {
		t.Errorf("restart: %q", stdout)
	}
	if got := messages(f.bodies()[2]); len(got) != 5 || got[4] != "user: Third" {
		t.Errorf("third request %q", got)
	}
}

func TestInteractiveMissingKey(t *testing.T) {
	isolateEnv(t)
	for _, args := range [][]string{{}, {"--plain", "--provider", "openai"}, {"--model", "gpt-5"}} {
		code, stdout, stderr := chatRun(t, "Hi\n", args...)
		if code != 3 || stdout != "" || stderr != "neurobrau: error: OPENAI_API_KEY is not set. Export it in your environment and try again.\n" {
			t.Errorf("%v: code=%d stdout=%q stderr=%q", args, code, stdout, stderr)
		}
	}
}

func TestInteractiveRestoresActiveSelection(t *testing.T) {
	f := startFakeOllama(t, tags, respond(200, answer))
	paths := statePaths(t)
	writeTestFile(t, paths.Active, fixture(t, "active.json")) // ollama, qwen3.5:4b
	writeTestFile(t, filepath.Join(paths.Conversations, "0b6f8e52-2c1d-4c7e-9a3b-5d2e1f0a9c41.json"), fixture(t, "conversation_ollama_unicode.json"))
	writeTestFile(t, filepath.Join(paths.Conversations, "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d.json"), fixture(t, "conversation_empty_after_reset.json"))
	writeTestFile(t, filepath.Join(paths.Conversations, "bad.json"), "{")
	before := tree(t, filepath.Dir(paths.Conversations))

	// Without input nothing is rewritten.
	code, stdout, stderr := chatRun(t, "")
	if code != ExitOK || !strings.Contains(stdout, "Provider: ollama | Model: qwen3.5:4b\n") ||
		!strings.Contains(stdout, "(4 messages). Type /reset to start a new one.") {
		t.Errorf("stdout %q", stdout)
	}
	badPath := filepath.Join(paths.Conversations, "bad.json")
	if !strings.HasPrefix(stderr, "warning: Ignoring unreadable conversation file "+badPath+": ") || strings.Count(stderr, "\n") != 1 {
		t.Errorf("stderr %q", stderr)
	}
	if !reflect.DeepEqual(before, tree(t, filepath.Dir(paths.Conversations))) {
		t.Error("starting the chat changed files")
	}

	chatRun(t, "Next\n")
	got := messages(f.chats[0])
	if len(got) != 5 || !strings.HasPrefix(got[0], "user: Grüße, 世界!") || got[4] != "user: Next" || f.chats[0]["model"] != "qwen3.5:4b" {
		t.Errorf("request %q", got)
	}
	if data, _ := os.ReadFile(badPath); string(data) != "{" {
		t.Error("corrupt file changed")
	}
}

func TestInteractiveSelectionFlags(t *testing.T) {
	f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
	paths := statePaths(t)
	writeTestFile(t, filepath.Join(paths.Conversations, "0b6f8e52-2c1d-4c7e-9a3b-5d2e1f0a9c41.json"), fixture(t, "conversation_ollama_unicode.json"))
	writeTestFile(t, filepath.Join(paths.Conversations, "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f.json"), fixture(t, "conversation_openai_usage.json"))
	writeTestFile(t, filepath.Join(paths.Conversations, "3f2a1b0c-8d7e-4f6a-9b5c-4d3e2f1a0b9c.json"), fixture(t, "conversation_legacy_no_provider.json"))
	// An active.json with an unknown provider is ignored with a warning.
	writeTestFile(t, paths.Active, `{"schema_version": 1, "provider": "anthropic", "model": "claude"}`)

	tests := []struct {
		args       []string
		wantBanner string
		wantFirst  string // first message sent
	}{
		{nil, "Provider: openai | Model: gpt-6-luna", "user: What is a neural network?"},
		{[]string{"--provider", "openai"}, "Provider: openai | Model: gpt-6-luna", "user: What is a neural network?"},
		// --model alone means OpenAI, even though an Ollama conversation of
		// that model exists; the legacy file matches any OpenAI model.
		{[]string{"--model", "qwen3.5:4b"}, "Provider: openai | Model: qwen3.5:4b", "user: Hello"},
	}
	for i, tt := range tests {
		code, stdout, stderr := chatRun(t, "Q\n", tt.args...)
		wantStderr := ""
		if i == 0 { // later runs find the active.json the first one saved
			wantStderr = "warning: Ignoring invalid " + paths.Active + ".\n"
		}
		if code != ExitOK || !strings.Contains(stdout, tt.wantBanner) || stderr != wantStderr {
			t.Errorf("%v: code=%d stdout=%q stderr=%q", tt.args, code, stdout, stderr)
		}
		bodies := f.bodies()
		if got := messages(bodies[len(bodies)-1]); got[0] != tt.wantFirst {
			t.Errorf("%v: sent %q", tt.args, got)
		}
	}
}

func TestInteractiveSavedSettings(t *testing.T) {
	const settingsJSON = `{"schema_version": 1, "answer_style": "brief", "providers": {"openai": {"models": {
	  "gpt-6-luna": {"reasoning": "low", "max_output_tokens": 300}}}}}`
	tests := []struct {
		name  string
		args  []string
		check map[string]any
	}{
		{"saved", nil, map[string]any{"reasoning": effort("low"), "max_output_tokens": float64(300), "instructions": llmStyle("brief")}},
		{"CLI over saved", []string{"--reasoning-effort", "high", "--max-output-tokens", "50", "--answer-style", "normal"},
			map[string]any{"reasoning": effort("high"), "max_output_tokens": float64(50), "instructions": nil}},
		{"model-default", []string{"--reasoning-effort", "model-default"}, map[string]any{"reasoning": nil, "max_output_tokens": float64(300)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
			paths := statePaths(t)
			writeTestFile(t, paths.Settings, settingsJSON)
			code, _, stderr := chatRun(t, "Hi\n", append([]string{"--no-history"}, tt.args...)...)
			if code != ExitOK || stderr != "" {
				t.Fatalf("code=%d stderr=%q", code, stderr)
			}
			body := f.bodies()[0]
			for k, v := range tt.check {
				if !reflect.DeepEqual(body[k], v) {
					t.Errorf("%s = %v, want %v", k, body[k], v)
				}
			}
			if data, _ := os.ReadFile(paths.Settings); string(data) != settingsJSON {
				t.Error("settings.json changed")
			}
		})
	}
}

func llmStyle(style string) any {
	switch style {
	case "brief":
		return "Answer briefly and concisely. Use a few sentences or a short list. " +
			"Leave out background and caveats unless the user asks for them."
	}
	return nil
}

func TestInteractiveInvalidSettings(t *testing.T) {
	t.Run("unsupported saved values", func(t *testing.T) {
		f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
		writeTestFile(t, statePaths(t).Settings, `{"schema_version": 1, "providers": {"openai": {"models": {
		  "gpt-6-luna": {"reasoning": "on"}, "gpt-4o": {"reasoning": "high", "max_output_tokens": 16385}}}}}`)
		code, _, stderr := chatRun(t, "Hi\n", "--no-history")
		if code != ExitOK || !strings.HasPrefix(stderr, "warning: Saved reasoning 'On' is not available for gpt-6-luna (gpt-6-luna supports reasoning: None, ") ||
			!strings.HasSuffix(stderr, "; using None.\n") {
			t.Errorf("code=%d stderr=%q", code, stderr)
		}
		code, _, stderr = chatRun(t, "Hi\n", "--no-history", "--model", "gpt-4o")
		if code != ExitOK || stderr != "warning: Saved reasoning 'High' is not available for gpt-4o (gpt-4o does not support reasoning); using Model default.\n"+
			"warning: Saved setting ignored: 16385 exceeds the limit of 16384 output tokens for gpt-4o. Using the default.\n" {
			t.Errorf("code=%d stderr=%q", code, stderr)
		}
		if body := f.bodies()[0]; !reflect.DeepEqual(body["reasoning"], effort("none")) || body["max_output_tokens"] != nil {
			t.Errorf("request %v", body)
		}
	})
	t.Run("corrupt file is never overwritten", func(t *testing.T) {
		startFakeOpenAI(t, openAIReply(200, openAIAnswer))
		paths := statePaths(t)
		writeTestFile(t, paths.Settings, `{"schema_version": 1, "answer_style": `)
		code, _, stderr := chatRun(t, "Hi\n/reset\n", "--answer-style", "brief", "--reasoning-effort", "low")
		if code != ExitOK || !strings.HasPrefix(stderr, "warning: Ignoring invalid settings file "+paths.Settings+": ") {
			t.Errorf("code=%d stderr=%q", code, stderr)
		}
		if data, _ := os.ReadFile(paths.Settings); string(data) != `{"schema_version": 1, "answer_style": ` {
			t.Error("settings.json changed")
		}
	})
	t.Run("unsupported CLI value exits 2 before the chat", func(t *testing.T) {
		f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
		code, stdout, stderr := chatRun(t, "Hi\n", "--reasoning-effort", "minimal")
		if code != ExitUsage || stdout != "" || !strings.HasPrefix(stderr, "neurobrau: error: gpt-6-luna supports reasoning: ") {
			t.Errorf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
		}
		if len(f.models()) != 0 {
			t.Error("request sent")
		}
	})
	t.Run("Ollama model-default on an on/off model", func(t *testing.T) {
		startFakeOllamaWithShow(t, thinkingTags, thinkingShows, respond(200, answer))
		code, stdout, stderr := chatRun(t, "Hi\n", "--provider", "ollama", "--model", "qwen3.5:4b", "--reasoning-effort", "model-default")
		if code != ExitUsage || stdout != "" || stderr != "neurobrau: error: qwen3.5:4b has no model-default reasoning choice (qwen3.5:4b supports reasoning: Off, On).\n" {
			t.Errorf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
		}
	})
}

func TestInteractiveOllamaSavedThinking(t *testing.T) {
	f := startFakeOllamaWithShow(t, thinkingTags, thinkingShows, respond(200, answer))
	writeTestFile(t, statePaths(t).Settings, `{"schema_version": 1, "providers": {"ollama": {"models": {
	  "qwen3.5:4b": {"reasoning": "on", "max_output_tokens": 600}}}}}`)
	code, stdout, stderr := chatRun(t, "Hi\n", "--provider", "ollama", "--no-history")
	if code != ExitOK || stderr != "" || !strings.Contains(stdout, "Provider: ollama | Model: qwen3.5:4b") {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
	if body := f.chats[0]; body["think"] != true || !reflect.DeepEqual(body["options"], map[string]any{"num_predict": float64(600)}) {
		t.Errorf("request %v", body)
	}
}

func TestNoHistoryTouchesNothing(t *testing.T) {
	f := startFakeOpenAI(t, openAIReply(200, openAIAnswer))
	paths := statePaths(t)
	state := filepath.Dir(paths.Conversations)
	writeTestFile(t, paths.Active, `{"schema_version": 1, "provider": "openai", "model": "gpt-5"}`)
	writeTestFile(t, filepath.Join(paths.Conversations, "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f.json"), fixture(t, "conversation_openai_usage.json"))
	os.Chmod(state, 0o755)               // would be tightened to 0700 by any history access
	os.Chmod(paths.Conversations, 0o311) // unreadable: any read would warn
	t.Cleanup(func() { os.Chmod(paths.Conversations, 0o700) })
	before := tree(t, state)

	code, stdout, stderr := chatRun(t, "Hi\n/reset\nAgain\n/cleanup\n/exit\n", "--no-history")
	if code != ExitOK || stderr != "" {
		t.Errorf("code=%d stderr=%q", code, stderr)
	}
	for _, want := range []string{chatBanner, "Conversation history cleared.\n", "Conversation history is disabled for this session (--no-history).\n"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("stdout lacks %q: %q", want, stdout)
		}
	}
	if strings.Contains(stdout, "Restored") {
		t.Error("a conversation was restored")
	}
	if bodies := f.bodies(); len(bodies) != 2 || len(messages(bodies[1])) != 1 {
		t.Errorf("after /reset the context was not empty: %v", bodies)
	}
	if after := tree(t, state); !reflect.DeepEqual(before, after) {
		t.Errorf("--no-history changed the state directory:\n%v\n%v", before, after)
	}
}

// lockedBuffer is a stream the test can read while the CLI writes to it.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func TestInteractiveInterruptWhileIdle(t *testing.T) {
	startFakeOpenAI(t, openAIReply(200, openAIAnswer))
	in, w := io.Pipe() // nothing is ever typed
	defer w.Close()
	interrupts := make(chan os.Signal, 1)
	var out, errOut lockedBuffer
	done := make(chan int)
	go func() {
		done <- Run(context.Background(), nil, Streams{Stdin: in, Stdout: &out, Stderr: &errOut, Interrupts: interrupts})
	}()
	for !strings.HasSuffix(out.String(), "YOU > ") { // the chat is waiting for input
		runtime.Gosched()
	}
	interrupts <- os.Interrupt
	if code := <-done; code != ExitInterrupted || out.String() != chatBanner+"YOU > \nGoodbye.\n" || errOut.String() != "" {
		t.Errorf("code=%d stdout=%q stderr=%q", code, out.String(), errOut.String())
	}
}

func TestInteractiveInterruptDuringStartup(t *testing.T) {
	interrupts := make(chan os.Signal, 1)
	isolateEnv(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		interrupts <- os.Interrupt // Ctrl+C while the default model is looked up
		<-r.Context().Done()
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	var out, errOut bytes.Buffer
	code := Run(context.Background(), []string{"--provider", "ollama"}, Streams{Stdin: strings.NewReader("Hi\n"), Stdout: &out, Stderr: &errOut, Interrupts: interrupts})
	if code != ExitInterrupted || out.String() != "" || errOut.String() != "\nneurobrau: interrupted\n" {
		t.Errorf("code=%d stdout=%q stderr=%q", code, out.String(), errOut.String())
	}
}

func TestOneShotInterruptSignal(t *testing.T) {
	interrupts := make(chan os.Signal, 1)
	startFakeOllama(t, tags, func(w http.ResponseWriter, r *http.Request) {
		interrupts <- os.Interrupt // Ctrl+C while the model is generating
		<-r.Context().Done()
	})
	var out, errOut bytes.Buffer
	code := Run(context.Background(), []string{"--provider", "ollama", "Hi"}, Streams{Stdout: &out, Stderr: &errOut, Interrupts: interrupts})
	if code != ExitInterrupted || out.String() != "" || errOut.String() != "\nneurobrau: interrupted\n" {
		t.Errorf("code=%d stdout=%q stderr=%q", code, out.String(), errOut.String())
	}
}
