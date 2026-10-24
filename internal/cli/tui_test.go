package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"

	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// fakeTerminal makes the given streams count as terminals and hands out
// simulated screens that type keys: typed is entered once the UI runs,
// then Ctrl+D is pressed until it exits (it is ignored while an answer
// is being generated).
type fakeTerminal struct {
	mu      sync.Mutex
	screens int
	finis   int
	typed   string
	// script, if set, runs first; it drives the UI with keys and waits.
	script   func(s *scriptedScreen)
	failures []string // waits that timed out in the script
	failure  error    // newScreen fails with this
}

func useFakeTerminal(t *testing.T, ft *fakeTerminal, terminals ...any) {
	t.Helper()
	oldIsTerminal, oldNewScreen := isTerminal, newScreen
	t.Cleanup(func() { isTerminal, newScreen = oldIsTerminal, oldNewScreen })
	isTerminal = func(stream any) bool {
		for _, s := range terminals {
			if s == stream {
				return true
			}
		}
		return false
	}
	newScreen = func() (tcell.Screen, error) {
		if ft.failure != nil {
			return nil, ft.failure
		}
		ft.mu.Lock()
		ft.screens++
		ft.mu.Unlock()
		return &scriptedScreen{SimulationScreen: tcell.NewSimulationScreen("UTF-8"), ft: ft, done: make(chan struct{})}, nil
	}
}

func (ft *fakeTerminal) counts() (screens, finis int) {
	ft.mu.Lock()
	defer ft.mu.Unlock()
	return ft.screens, ft.finis
}

type scriptedScreen struct {
	tcell.SimulationScreen
	ft     *fakeTerminal
	done   chan struct{}
	once   sync.Once
	drawMu sync.Mutex // orders redraws with reading the cells
}

func (s *scriptedScreen) Show() {
	s.drawMu.Lock()
	defer s.drawMu.Unlock()
	s.SimulationScreen.Show()
}

func (s *scriptedScreen) Sync() {
	s.drawMu.Lock()
	defer s.drawMu.Unlock()
	s.SimulationScreen.Sync()
}

// text is the screen, one line per row.
func (s *scriptedScreen) text() string {
	s.drawMu.Lock()
	defer s.drawMu.Unlock()
	cells, w, _ := s.GetContents()
	var b strings.Builder
	for i, c := range cells {
		if i > 0 && i%w == 0 {
			b.WriteByte('\n')
		}
		if len(c.Runes) > 0 {
			b.WriteRune(c.Runes[0])
		}
	}
	return b.String()
}

// waitText waits until text is on the screen; a timeout is recorded.
func (s *scriptedScreen) waitText(text string) {
	deadline := time.Now().Add(5 * time.Second)
	for !strings.Contains(s.text(), text) {
		if time.Now().After(deadline) {
			s.ft.mu.Lock()
			s.ft.failures = append(s.ft.failures, fmt.Sprintf("no %q on\n%s", text, s.text()))
			s.ft.mu.Unlock()
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
}

func (s *scriptedScreen) keys(keys ...tcell.Key) {
	for _, k := range keys {
		s.InjectKey(k, 0, tcell.ModNone)
	}
}

func (s *scriptedScreen) typeLine(text string) {
	for _, r := range text {
		s.InjectKey(tcell.KeyRune, r, tcell.ModNone)
	}
	s.InjectKey(tcell.KeyEnter, 0, tcell.ModNone)
}

func (s *scriptedScreen) Init() error {
	if err := s.SimulationScreen.Init(); err != nil {
		return err
	}
	s.SetSize(80, 25)
	go func() {
		if s.ft.script != nil {
			s.ft.script(s)
		}
		for _, r := range s.ft.typed {
			if r == '\n' {
				s.InjectKey(tcell.KeyEnter, 0, tcell.ModNone)
			} else {
				s.InjectKey(tcell.KeyRune, r, tcell.ModNone)
			}
		}
		for {
			s.InjectKey(tcell.KeyCtrlD, 0, tcell.ModNone)
			select {
			case <-s.done:
				return
			case <-time.After(10 * time.Millisecond):
			}
		}
	}()
	return nil
}

func (s *scriptedScreen) Fini() {
	s.once.Do(func() {
		s.ft.mu.Lock()
		s.ft.finis++
		s.ft.mu.Unlock()
		close(s.done)
	})
	s.SimulationScreen.Fini()
}

// ttyRun runs the CLI with stdin and stdout as fake terminals.
func ttyRun(t *testing.T, ft *fakeTerminal, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	stdin := strings.NewReader("")
	var out, errOut bytes.Buffer
	useFakeTerminal(t, ft, stdin, &out)
	code = Run(context.Background(), args, Streams{Stdin: stdin, Stdout: &out, Stderr: &errOut})
	return code, out.String(), errOut.String()
}

func TestStartupRouting(t *testing.T) {
	tests := []struct {
		name       string
		args       []string
		term       string
		stdinTTY   bool
		stdoutTTY  bool
		fullscreen bool
	}{
		{"terminal", nil, "xterm-256color", true, true, true},
		{"terminal with --stream", []string{"--stream"}, "xterm", true, true, true},
		{"--plain", []string{"--plain"}, "xterm", true, true, false},
		{"--plain --stream", []string{"--plain", "--stream"}, "xterm", true, true, false},
		{"TERM unset", nil, "", true, true, false},
		{"TERM dumb", nil, "dumb", true, true, false},
		{"stdin redirected", nil, "xterm", false, true, false},
		{"stdout redirected", nil, "xterm", true, false, false},
		{"prompt", []string{"--stream", "Hi"}, "xterm", true, true, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startFakeOllamaWithShow(t, tags, thinkingShows, chunks(ollamaLine("Hello!"), ollamaDone))
			t.Setenv("TERM", tt.term)
			stdin := strings.NewReader("")
			var out, errOut bytes.Buffer
			var terminals []any
			if tt.stdinTTY {
				terminals = append(terminals, stdin)
			}
			if tt.stdoutTTY {
				terminals = append(terminals, &out)
			}
			ft := &fakeTerminal{}
			useFakeTerminal(t, ft, terminals...)
			args := append([]string{"--provider", "ollama", "--model", "qwen3.5:4b"}, tt.args...)
			code := Run(context.Background(), args, Streams{Stdin: stdin, Stdout: &out, Stderr: &errOut})
			screens, finis := ft.counts()
			if code != ExitOK || screens != finis || (screens == 1) != tt.fullscreen {
				t.Errorf("code=%d screens=%d finis=%d stdout=%q stderr=%q", code, screens, finis, out.String(), errOut.String())
			}
			plain := strings.Contains(out.String(), "YOU > ")
			if tt.name == "prompt" {
				plain = out.String() == "Hello!\n" // one-shot
			}
			if tt.fullscreen == plain {
				t.Errorf("stdout %q", out.String())
			}
		})
	}
}

func TestFullscreenStartupErrorsLeaveTheTerminalAlone(t *testing.T) {
	t.Run("missing key", func(t *testing.T) {
		isolateEnv(t)
		t.Setenv("TERM", "xterm")
		ft := &fakeTerminal{}
		code, stdout, stderr := ttyRun(t, ft)
		if screens, _ := ft.counts(); code != 3 || screens != 0 || stdout != "" || !strings.Contains(stderr, "OPENAI_API_KEY") {
			t.Errorf("code=%d screens=%d stdout=%q stderr=%q", code, screens, stdout, stderr)
		}
	})
	t.Run("Ollama unreachable", func(t *testing.T) {
		isolateEnv(t)
		t.Setenv("TERM", "xterm")
		ft := &fakeTerminal{}
		code, _, _ := ttyRun(t, ft, "--provider", "ollama")
		if screens, _ := ft.counts(); code != 7 || screens != 0 {
			t.Errorf("code=%d screens=%d", code, screens)
		}
	})
	t.Run("unsupported setting", func(t *testing.T) {
		startFakeOllamaWithShow(t, tags, thinkingShows, respond(500, ""))
		t.Setenv("TERM", "xterm")
		ft := &fakeTerminal{}
		code, _, stderr := ttyRun(t, ft, "--provider", "ollama", "--model", "llama3.2:3b", "--reasoning-effort", "high")
		if screens, _ := ft.counts(); code != ExitUsage || screens != 0 {
			t.Errorf("code=%d screens=%d stderr=%q", code, screens, stderr)
		}
	})
	t.Run("screen failure", func(t *testing.T) {
		startFakeOllamaWithShow(t, tags, thinkingShows, respond(500, ""))
		t.Setenv("TERM", "xterm")
		ft := &fakeTerminal{failure: errors.New("no terminfo")}
		code, stdout, stderr := ttyRun(t, ft, "--provider", "ollama")
		if code != 1 || stdout != "" ||
			stderr != "neurobrau: error: cannot start the fullscreen UI: no terminfo (use --plain for the line-oriented chat)\n" {
			t.Errorf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
		}
	})
}

func TestFullscreenConversation(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := map[bool]string{false: "complete answers", true: "--stream"}[stream]
		t.Run(name, func(t *testing.T) {
			handler := respond(200, `{"model":"qwen3.5:4b","message":{"role":"assistant","content":"Hello!"},"done":true,"prompt_eval_count":3,"eval_count":2}`)
			if stream {
				handler = chunks(ollamaLine("Hel"), ollamaLine("lo!"), ollamaDone)
			}
			f := startFakeOllamaWithShow(t, tags, thinkingShows, sequence(handler, handler))
			t.Setenv("TERM", "xterm")
			paths := statePaths(t)
			const settingsJSON = `{"schema_version": 1, "answer_style": "brief"}`
			writeTestFile(t, paths.Settings, settingsJSON)
			args := []string{"--provider", "ollama", "--model", "qwen3.5:4b", "--transcript-on-exit"}
			if stream {
				args = append(args, "--stream")
			}
			ft := &fakeTerminal{typed: "Hi\n"}
			code, stdout, stderr := ttyRun(t, ft, args...)
			if code != ExitOK || stderr != "" || stdout != "YOU > Hi\n\nAI > Hello!\n" {
				t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
			}
			body := f.chats[0]
			if body["stream"] != stream || !reflect.DeepEqual(messages(body), []string{"user: Hi"}) {
				t.Errorf("request %v", body)
			}
			if sys := body["messages"].([]any)[0].(map[string]any); sys["role"] != "system" || sys["content"] != llmStyle("brief") {
				t.Errorf("answer style not applied: %v", sys)
			}
			if data, _ := os.ReadFile(paths.Settings); string(data) != settingsJSON {
				t.Error("settings.json changed")
			}
			files, _ := filepath.Glob(filepath.Join(paths.Conversations, "*.json"))
			if len(files) != 1 {
				t.Fatalf("files %v", files)
			}
			c, err := storage.NewHistoryStore(paths.Conversations).Load(files[0])
			if err != nil || len(c.Messages) != 2 || c.Messages[1].Content != "Hello!" || len(c.Requests) != 1 {
				t.Errorf("saved %+v %v", c, err)
			}

			// The next start restores it and sends it as context.
			ft = &fakeTerminal{typed: "Again\n"}
			if code, stdout, _ := ttyRun(t, ft, args...); code != ExitOK || !strings.HasSuffix(stdout, "YOU > Again\n\nAI > Hello!\n") {
				t.Errorf("restart: code=%d stdout=%q", code, stdout)
			}
			if got := messages(f.chats[1]); !reflect.DeepEqual(got, []string{"user: Hi", "assistant: Hello!", "user: Again"}) {
				t.Errorf("restored context %q", got)
			}
		})
	}
}

func TestFullscreenNoHistory(t *testing.T) {
	startFakeOllamaWithShow(t, tags, thinkingShows, chunks(ollamaLine("A1"), ollamaDone))
	t.Setenv("TERM", "xterm")
	paths := statePaths(t)
	state := filepath.Dir(paths.Conversations)
	writeTestFile(t, filepath.Join(paths.Conversations, "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f.json"), fixture(t, "conversation_openai_usage.json"))
	os.Chmod(state, 0o755)               // any history access would tighten it to 0700
	os.Chmod(paths.Conversations, 0o311) // unreadable: any read would warn
	t.Cleanup(func() { os.Chmod(paths.Conversations, 0o700) })
	before := tree(t, state)

	ft := &fakeTerminal{typed: "Q1\n"}
	code, stdout, stderr := ttyRun(t, ft, "--no-history", "--stream", "--provider", "ollama", "--model", "llama3.2:3b", "--transcript-on-exit")
	if code != ExitOK || stderr != "" || stdout != "YOU > Q1\n\nAI > A1\n" {
		t.Errorf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
	if after := tree(t, state); !reflect.DeepEqual(before, after) {
		t.Errorf("--no-history changed the state directory")
	}
}

// fakeServices is one httptest server for Ollama (/api/…) and the OpenAI
// API (/v1/…), with the environment pointed at it.
type fakeServices struct {
	mu        sync.Mutex
	shows     map[string]int   // /api/show calls per model
	chats     []map[string]any // /api/chat bodies
	responses []map[string]any // /v1/responses bodies
	modelList int              // /v1/models calls
}

const servicesTags = `{"models":[{"model":"glm-4.7:cloud","capabilities":["completion"]},` +
	`{"model":"qwen3.5:4b","capabilities":["completion","thinking"],"details":{"parameter_size":"4.7B","quantization_level":"Q4_K_M"}},` +
	`{"model":"llama3.2:3b","capabilities":["completion"],"details":{"parameter_size":"3.2B","quantization_level":"Q4_K_M"}}]}`

func startFakeServices(t *testing.T, openAIKey bool) *fakeServices {
	t.Helper()
	isolateEnv(t)
	f := &fakeServices{shows: map[string]int{}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		defer f.mu.Unlock()
		switch r.Method + " " + r.URL.Path {
		case "GET /api/tags":
			io.WriteString(w, servicesTags)
		case "POST /api/show":
			f.shows[body["model"].(string)]++
			io.WriteString(w, thinkingShows[body["model"].(string)])
		case "POST /api/chat":
			f.chats = append(f.chats, body)
			io.WriteString(w, `{"model":"x","message":{"role":"assistant","content":"Hello from Ollama"},"done":true,"prompt_eval_count":3,"eval_count":2}`)
		case "GET /v1/models":
			f.modelList++
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"object":"list","data":[{"id":"gpt-6-luna","object":"model","created":0,"owned_by":"openai"},`+
				`{"id":"text-embedding-3-small","object":"model","created":0,"owned_by":"openai"}]}`)
		case "POST /v1/responses":
			f.responses = append(f.responses, body)
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, strings.ReplaceAll(openAIAnswer, "\n", ""))
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)
	t.Setenv("OPENAI_BASE_URL", srv.URL+"/v1")
	if openAIKey {
		t.Setenv("OPENAI_API_KEY", fakeKey)
	}
	t.Setenv("TERM", "xterm")
	return f
}

func (ft *fakeTerminal) check(t *testing.T) {
	t.Helper()
	ft.mu.Lock()
	defer ft.mu.Unlock()
	for _, f := range ft.failures {
		t.Error(f)
	}
}

// TestFullscreenMenus drives F2–F4 with the real Ollama and OpenAI
// providers against fake servers.
func TestFullscreenMenus(t *testing.T) {
	f := startFakeServices(t, false)
	paths := statePaths(t)
	writeTestFile(t, paths.Settings, `{"schema_version": 1, "answer_style": "normal", "providers": {"ollama": {"models": {"qwen3.5:4b": {"reasoning": "on"}}}}}`)
	args := []string{"--provider", "ollama", "--model", "qwen3.5:4b", "--transcript-on-exit"}
	ft := &fakeTerminal{script: func(s *scriptedScreen) {
		s.waitText("|  on ") // the saved choice beats the built-in Off
		s.keys(tcell.KeyF3)
		s.waitText("SELECT PROVIDER")
		s.keys(tcell.KeyUp, tcell.KeyEnter)
		s.waitText("OpenAI is not available: OPENAI_API_KEY is not set.")
		s.keys(tcell.KeyF2)
		s.waitText("SELECT OLLAMA MODEL")
		s.waitText("* qwen3.5:4b  ")
		s.waitText("3.2B Q4_K_M")
		if strings.Contains(s.text(), "glm-4.7") {
			s.waitText("cloud models must not be listed")
		}
		s.keys(tcell.KeyEsc, tcell.KeyF4)
		s.waitText("REASONING ]")
		s.keys(tcell.KeyHome, tcell.KeyEnter)
		s.waitText("Reasoning for qwen3.5:4b: Off. Applies from")
		s.typeLine("Hi")
		s.waitText("AI > Hello from Ollama")
	}}
	code, stdout, stderr := ttyRun(t, ft, args...)
	ft.check(t)
	if code != ExitOK || stderr != "" || stdout != "YOU > Hi\n\nAI > Hello from Ollama\n" {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
	if f.chats[0]["think"] != false {
		t.Errorf("think = %v", f.chats[0]["think"])
	}
	if f.shows["qwen3.5:4b"] != 1 {
		t.Errorf("/api/show requests: %v (cached after the first)", f.shows)
	}
	saved, _ := os.ReadFile(paths.Settings)
	settings, _, err := storage.DecodeSettings(saved)
	if err != nil || settings.ForModel("ollama", "qwen3.5:4b").Reasoning != "off" {
		t.Errorf("settings %s", saved)
	}

	// The next start uses the saved Off without writing anything.
	ft = &fakeTerminal{typed: "Again\n"}
	if code, _, _ := ttyRun(t, ft, args...); code != ExitOK {
		t.Fatalf("restart: code %d", code)
	}
	if f.chats[1]["think"] != false {
		t.Errorf("after restart think = %v", f.chats[1]["think"])
	}
	if again, _ := os.ReadFile(paths.Settings); string(again) != string(saved) {
		t.Error("settings.json written without a change")
	}
}

func TestFullscreenSwitchProviders(t *testing.T) {
	f := startFakeServices(t, true)
	paths := statePaths(t)
	ft := &fakeTerminal{script: func(s *scriptedScreen) {
		s.typeLine("Ollama question")
		s.waitText("AI > Hello from Ollama")
		s.keys(tcell.KeyF3)
		s.waitText("SELECT PROVIDER")
		s.keys(tcell.KeyUp, tcell.KeyEnter)
		s.waitText("Now using OpenAI / gpt-6-luna. Started a new")
		s.keys(tcell.KeyF6)
		s.waitText("ANSWER STYLE ]")
		s.keys(tcell.KeyHome, tcell.KeyEnter)
		s.waitText("Answer style: Brief.")
		s.typeLine("OpenAI question")
		s.waitText("AI > ")
		s.waitText(" in 5 out 7")
		s.keys(tcell.KeyF2)
		s.waitText("SELECT OPENAI MODEL")
		if strings.Contains(s.text(), "embedding") {
			s.waitText("non-text models must not be listed")
		}
		s.keys(tcell.KeyEsc, tcell.KeyF3)
		s.waitText("SELECT PROVIDER")
		s.keys(tcell.KeyDown, tcell.KeyEnter)
		s.waitText("conversation (2 messages).")
	}}
	code, _, stderr := ttyRun(t, ft, "--provider", "ollama", "--model", "qwen3.5:4b")
	ft.check(t)
	if code != ExitOK || stderr != "" {
		t.Fatalf("code=%d stderr=%q", code, stderr)
	}
	if len(f.responses) != 1 {
		t.Fatalf("responses %v", f.responses)
	}
	body := f.responses[0]
	if body["model"] != "gpt-6-luna" || body["instructions"] != llmStyle("brief") || len(messages(body)) != 1 {
		t.Errorf("OpenAI request %v", body)
	}
	if f.modelList != 1 {
		t.Errorf("/v1/models requests: %d (cached)", f.modelList)
	}
	files, _ := filepath.Glob(filepath.Join(paths.Conversations, "*.json"))
	if len(files) != 2 {
		t.Errorf("conversations %v", files)
	}
	active, _ := os.ReadFile(paths.Active)
	if !strings.Contains(string(active), `"ollama"`) {
		t.Errorf("active.json %s", active)
	}
	settings, _ := os.ReadFile(paths.Settings)
	if !strings.Contains(string(settings), `"answer_style": "brief"`) {
		t.Errorf("settings %s", settings)
	}
}
