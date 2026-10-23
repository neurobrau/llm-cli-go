package main

import (
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
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
			cmd.Env = append(isolatedEnv(), runMainEnv+"=1")
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
// with both servers pointed at a closed port, so a subprocess can never
// use real credentials or reach a real service.
func isolatedEnv() []string {
	var env []string
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "OPENAI_") && !strings.HasPrefix(kv, "OLLAMA_") {
			env = append(env, kv)
		}
	}
	return append(env, "OPENAI_BASE_URL=http://127.0.0.1:1/v1", "OLLAMA_HOST=127.0.0.1:1")
}
