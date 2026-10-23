package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// When this variable is set, the test binary behaves like the real
// executable, so the tests below can observe real process exit codes.
const runMainEnv = "NEUROBRAU_TEST_RUN_MAIN"

func TestMain(m *testing.M) {
	if os.Getenv(runMainEnv) == "1" {
		main()
	}
	os.Exit(m.Run())
}

func TestMainExitCodes(t *testing.T) {
	tests := []struct {
		args     []string
		wantCode int
	}{
		{[]string{"--version"}, 0},
		{[]string{"--help"}, 0},
		{[]string{"--not-a-real-flag"}, 2},
		{[]string{"two", "prompts"}, 2},
		{[]string{"Hi"}, 3}, // OpenAI default without OPENAI_API_KEY
	}
	for _, tt := range tests {
		t.Run(tt.args[0], func(t *testing.T) {
			cmd := exec.Command(os.Args[0], tt.args...)
			cmd.Env = append(isolatedEnv(t), runMainEnv+"=1")
			err := cmd.Run()

			code := 0
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) {
				code = exitErr.ExitCode()
			} else if err != nil {
				t.Fatalf("running %v: %v", tt.args, err)
			}
			if code != tt.wantCode {
				t.Errorf("exit code = %d, want %d", code, tt.wantCode)
			}
		})
	}
}

// isolatedEnv is the current environment without OpenAI or Ollama settings,
// with both servers pointed at a closed port and HOME and the XDG
// directories in a temp directory, so a subprocess can never use real
// credentials, reach a real service or touch real user files.
func isolatedEnv(t *testing.T) []string {
	var env []string
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "OPENAI_") && !strings.HasPrefix(kv, "OLLAMA_") &&
			!strings.HasPrefix(kv, "HOME=") && !strings.HasPrefix(kv, "XDG_") {
			env = append(env, kv)
		}
	}
	home := t.TempDir()
	return append(env, "OPENAI_BASE_URL=http://127.0.0.1:1/v1", "OLLAMA_HOST=127.0.0.1:1",
		"HOME="+home, "XDG_CONFIG_HOME="+filepath.Join(home, "config"), "XDG_STATE_HOME="+filepath.Join(home, "state"))
}

// fakeOpenAI serves the Responses API; answer decides each reply.
func fakeOpenAI(t *testing.T, answer func(w http.ResponseWriter, r *http.Request, input []any)) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct{ Input []any }
		json.NewDecoder(r.Body).Decode(&body)
		answer(w, r, body.Input)
	}))
	t.Cleanup(srv.Close)
	return srv
}

const reply = `{"id":"r","status":"completed","model":"gpt-6-luna","output":[{"type":"message","role":"assistant",
  "content":[{"type":"output_text","text":"Answer %d"}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}`

// chatCommand runs the executable as an interactive chat against srv.
func chatCommand(t *testing.T, srv *httptest.Server, args ...string) (*exec.Cmd, string) {
	cmd := exec.Command(os.Args[0], args...)
	cmd.Env = append(isolatedEnv(t), runMainEnv+"=1", "OPENAI_BASE_URL="+srv.URL+"/v1",
		"OPENAI_API_KEY=sk-test-synthetic-key-0000")
	var state string
	for _, kv := range cmd.Env {
		if v, ok := strings.CutPrefix(kv, "XDG_STATE_HOME="); ok {
			state = v
		}
	}
	return cmd, filepath.Join(state, "neurobrau", "conversations")
}

func exitCode(t *testing.T, err error) int {
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode()
	}
	if err != nil {
		t.Fatal(err)
	}
	return 0
}

func TestPipedChat(t *testing.T) {
	var mu sync.Mutex
	turns := 0
	srv := fakeOpenAI(t, func(w http.ResponseWriter, r *http.Request, input []any) {
		mu.Lock()
		turns++
		n := turns
		mu.Unlock()
		if len(input) != 2*n-1 {
			t.Errorf("request %d has %d input messages", n, len(input))
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, strings.Replace(reply, "%d", string(rune('0'+n)), 1))
	})
	cmd, conversations := chatCommand(t, srv)
	cmd.Stdin = strings.NewReader("Hello\n/usage\nSecond\n")
	out, err := cmd.Output()
	if code := exitCode(t, err); code != 0 {
		t.Fatalf("exit code %d", code)
	}
	for _, want := range []string{"Provider: openai | Model: gpt-6-luna\n", "YOU > AI > Answer 1\n",
		"YOU > Conversation usage (1 request(s)", "YOU > AI > Answer 2\nYOU > \nGoodbye.\n"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("stdout lacks %q:\n%s", want, out)
		}
	}
	files, _ := filepath.Glob(filepath.Join(conversations, "*.json"))
	if len(files) != 1 {
		t.Fatalf("conversation files %v", files)
	}
	data, _ := os.ReadFile(files[0])
	if strings.Count(string(data), `"role"`) != 4 || strings.Count(string(data), `"status": "completed"`) != 2 {
		t.Errorf("saved conversation:\n%s", data)
	}
}

// TestCtrlC sends real SIGINTs: the first cancels the running request, the
// second, at the prompt, ends the chat with exit code 130.
func TestCtrlC(t *testing.T) {
	requests := make(chan struct{}, 1)
	srv := fakeOpenAI(t, func(w http.ResponseWriter, r *http.Request, input []any) {
		requests <- struct{}{}
		<-r.Context().Done() // never answers
	})
	cmd, conversations := chatCommand(t, srv, "--plain")
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer cmd.Process.Kill()
	out := bufio.NewReader(stdout)
	waitPrompt := func() {
		t.Helper()
		var seen string
		for !strings.HasSuffix(seen, "YOU > ") {
			b, err := out.ReadByte()
			if err != nil {
				t.Fatalf("reading stdout: %v (%q)", err, seen)
			}
			seen += string(b)
		}
	}
	waitPrompt()
	io.WriteString(stdin, "Question\n")
	select {
	case <-requests:
	case <-time.After(10 * time.Second):
		t.Fatal("no request")
	}
	cmd.Process.Signal(syscall.SIGINT)
	waitPrompt() // back at the prompt
	cmd.Process.Signal(syscall.SIGINT)
	rest, _ := io.ReadAll(out)
	if code := exitCode(t, cmd.Wait()); code != 130 {
		t.Errorf("exit code %d", code)
	}
	if string(rest) != "\nGoodbye.\n" || stderr.String() != "\nRequest cancelled.\n" {
		t.Errorf("stdout %q stderr %q", rest, stderr.String())
	}
	if files, _ := filepath.Glob(filepath.Join(conversations, "*.json")); len(files) != 0 {
		t.Errorf("canceled exchange saved: %v", files)
	}
}
