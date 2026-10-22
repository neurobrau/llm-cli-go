package cli

import (
	"bytes"
	"strings"
	"testing"
)

func run(t *testing.T, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errOut bytes.Buffer
	code = Run(args, &out, &errOut)
	return code, out.String(), errOut.String()
}

func TestHelp(t *testing.T) {
	for _, arg := range []string{"--help", "-h", "-help"} {
		t.Run(arg, func(t *testing.T) {
			code, stdout, stderr := run(t, arg)
			if code != ExitOK {
				t.Errorf("exit code = %d, want %d", code, ExitOK)
			}
			if !strings.HasPrefix(stdout, "usage: neurobrau ") {
				t.Errorf("stdout does not start with usage line:\n%s", stdout)
			}
			for _, want := range []string{"--help", "--version", "not implemented yet"} {
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
	code, stdout, stderr := run(t, "--help", "unexpected")
	if code != ExitOK || !strings.HasPrefix(stdout, "usage: ") || stderr != "" {
		t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

func TestNoArgumentsPrintsHelp(t *testing.T) {
	code, stdout, stderr := run(t)
	if code != ExitOK || !strings.HasPrefix(stdout, "usage: ") || stderr != "" {
		t.Errorf("got code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

func TestVersion(t *testing.T) {
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
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"unknown long flag", []string{"--not-a-real-flag"}, "not-a-real-flag"},
		{"unknown short flag", []string{"-x"}, "-x"},
		{"positional argument", []string{"unexpected-input"}, `unexpected argument: "unexpected-input"`},
		{"empty positional argument", []string{""}, `unexpected argument: ""`},
		{"positional after version", []string{"--version", "extra"}, `unexpected argument: "extra"`},
		{"positional after terminator", []string{"--", "--version"}, `unexpected argument: "--version"`},
		{"malformed boolean value", []string{"--version=maybe"}, "version"},
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
