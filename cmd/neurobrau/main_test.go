package main

import (
	"errors"
	"os"
	"os/exec"
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
		{[]string{"unexpected-input"}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.args[0], func(t *testing.T) {
			cmd := exec.Command(os.Args[0], tt.args...)
			cmd.Env = append(os.Environ(), runMainEnv+"=1")
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
