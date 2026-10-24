//go:build linux

package main

import (
	"errors"
	"fmt"
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
	"unsafe"
)

func ioctl(f *os.File, req uintptr, arg unsafe.Pointer) error {
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), req, uintptr(arg)); errno != 0 {
		return errno
	}
	return nil
}

// openPTY opens a pseudo-terminal pair of 80x25; skips the test where
// that is not possible.
func openPTY(t *testing.T) (master, slave *os.File) {
	t.Helper()
	master, err := os.OpenFile("/dev/ptmx", os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		t.Skipf("no pseudo-terminal: %v", err)
	}
	t.Cleanup(func() { master.Close() })
	var unlock int32
	var n uint32
	if err := ioctl(master, syscall.TIOCSPTLCK, unsafe.Pointer(&unlock)); err != nil {
		t.Skipf("unlockpt: %v", err)
	}
	if err := ioctl(master, syscall.TIOCGPTN, unsafe.Pointer(&n)); err != nil {
		t.Skipf("ptsname: %v", err)
	}
	slave, err = os.OpenFile(fmt.Sprintf("/dev/pts/%d", n), os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		t.Skipf("open slave: %v", err)
	}
	t.Cleanup(func() { slave.Close() })
	size := struct{ rows, cols, x, y uint16 }{25, 80, 0, 0}
	if err := ioctl(slave, syscall.TIOCSWINSZ, unsafe.Pointer(&size)); err != nil {
		t.Fatal(err)
	}
	return master, slave
}

func termios(t *testing.T, f *os.File) syscall.Termios {
	t.Helper()
	var tio syscall.Termios
	if err := ioctl(f, syscall.TCGETS, unsafe.Pointer(&tio)); err != nil {
		t.Fatal(err)
	}
	return tio
}

// ptyOutput collects what the process writes to the terminal.
type ptyOutput struct {
	mu  sync.Mutex
	buf strings.Builder
}

func (o *ptyOutput) String() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.buf.String()
}

func (o *ptyOutput) waitFor(t *testing.T, want string) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !strings.Contains(o.String(), want) {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %q in %q", want, o.String())
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// startFullscreen runs the executable on a pseudo-terminal, against a
// fake Ollama server, with isolated HOME and XDG directories.
func startFullscreen(t *testing.T, chat http.HandlerFunc, args ...string) (*exec.Cmd, *os.File, *ptyOutput, syscall.Termios, string) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			io.WriteString(w, `{"models":[{"model":"llama3.2:3b","capabilities":["completion"]}]}`)
		case "/api/show":
			io.WriteString(w, `{"capabilities":["completion"]}`)
		case "/api/chat":
			chat(w, r)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	master, slave := openPTY(t)
	before := termios(t, slave)

	cmd := exec.Command(os.Args[0], append([]string{"--provider", "ollama", "--model", "llama3.2:3b"}, args...)...)
	cmd.Env = append(isolatedEnv(t), runMainEnv+"=1", "OLLAMA_HOST="+srv.URL, "TERM=xterm-256color")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = slave, slave, slave
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true, Setctty: true, Ctty: 0}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	out := &ptyOutput{}
	go func() {
		buf := make([]byte, 4096)
		for {
			n, err := master.Read(buf)
			out.mu.Lock()
			out.buf.Write(buf[:n])
			out.mu.Unlock()
			if err != nil {
				return
			}
		}
	}()
	var state string
	for _, kv := range cmd.Env {
		if v, ok := strings.CutPrefix(kv, "XDG_STATE_HOME="); ok {
			state = v
		}
	}
	return cmd, master, out, before, filepath.Join(state, "neurobrau", "conversations")
}

func waitExit(t *testing.T, cmd *exec.Cmd) int {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		return exitCode(t, err)
	case <-time.After(10 * time.Second):
		cmd.Process.Kill()
		t.Fatal("the process did not exit")
		return -1
	}
}

// leaveAltScreen is xterm's sequence for leaving the alternate screen.
const leaveAltScreen = "\x1b[?1049l"

func TestFullscreenOnPTY(t *testing.T) {
	chat := func(w http.ResponseWriter, r *http.Request) {
		ollamaChunk(w, "Hello ")
		ollamaChunk(w, "there!")
		io.WriteString(w, `{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":3,"eval_count":2}`+"\n")
	}
	cmd, master, out, before, conversations := startFullscreen(t, chat, "--stream", "--transcript-on-exit")
	slave := cmd.Stdin.(*os.File)
	out.waitFor(t, "LLM SETTINGS")
	if termios(t, slave) == before {
		t.Error("the terminal is not in raw mode")
	}
	io.WriteString(master, "Hi\r")
	out.waitFor(t, "there!")
	out.waitFor(t, "DONE")
	io.WriteString(master, "\x04") // Ctrl+D
	if code := waitExit(t, cmd); code != 0 {
		t.Fatalf("exit code %d; output %q", code, out.String())
	}
	if after := termios(t, slave); after != before {
		t.Errorf("terminal settings not restored:\n%+v\n%+v", before, after)
	}
	out.waitFor(t, "AI > Hello there!\r\n")
	s := out.String()
	i := strings.LastIndex(s, leaveAltScreen)
	if i < 0 || !strings.HasSuffix(s[i:], "YOU > Hi\r\n\r\nAI > Hello there!\r\n") {
		t.Errorf("transcript is not printed after the terminal was restored: %q", s[max(0, i-20):])
	}
	if files, _ := filepath.Glob(filepath.Join(conversations, "*.json")); len(files) != 1 {
		t.Errorf("saved %v", files)
	}
}

func TestFullscreenCtrlCOnPTY(t *testing.T) {
	closed := make(chan struct{})
	chat := func(w http.ResponseWriter, r *http.Request) {
		ollamaChunk(w, "Partial")
		<-r.Context().Done() // stalls until the client closes the connection
		close(closed)
	}
	cmd, master, out, before, conversations := startFullscreen(t, chat, "--stream", "--transcript-on-exit")
	slave := cmd.Stdin.(*os.File)
	out.waitFor(t, "LLM SETTINGS")
	io.WriteString(master, "Hi\r")
	out.waitFor(t, "Partial")
	io.WriteString(master, "\x03") // Ctrl+C cancels the request
	out.waitFor(t, "CANCELLED")
	select {
	case <-closed:
	case <-time.After(10 * time.Second):
		t.Fatal("the connection was not closed")
	}
	io.WriteString(master, "\x03") // Ctrl+C while idle exits
	if code := waitExit(t, cmd); code != 130 {
		t.Fatalf("exit code %d", code)
	}
	if after := termios(t, slave); after != before {
		t.Error("terminal settings not restored")
	}
	s := out.String()
	if i := strings.LastIndex(s, leaveAltScreen); i < 0 || strings.Contains(s[i:], "YOU >") {
		t.Errorf("a canceled exchange was printed: %q", s[max(0, i):])
	}
	if files, _ := filepath.Glob(filepath.Join(conversations, "*.json")); len(files) != 0 {
		t.Errorf("canceled exchange saved: %v", files)
	}
}

// SIGTERM and SIGHUP end the fullscreen UI cooperatively: a running request
// is cancelled (its connection closed, nothing saved), the terminal is
// restored and the exit code is 128 + the signal number.
func TestFullscreenTerminationOnPTY(t *testing.T) {
	for _, tt := range []struct {
		sig      syscall.Signal
		code     int
		generate bool
	}{
		{syscall.SIGTERM, 143, true},
		{syscall.SIGHUP, 129, false},
	} {
		t.Run(tt.sig.String(), func(t *testing.T) {
			closed := make(chan struct{})
			chat := func(w http.ResponseWriter, r *http.Request) {
				ollamaChunk(w, "Partial")
				<-r.Context().Done()
				close(closed)
			}
			cmd, master, out, before, conversations := startFullscreen(t, chat, "--stream", "--transcript-on-exit")
			slave := cmd.Stdin.(*os.File)
			out.waitFor(t, "NeuroBräu Go CLI 0.4.0-dev")
			out.waitFor(t, "LLM SETTINGS")
			if tt.generate {
				io.WriteString(master, "Hi\r")
				out.waitFor(t, "Partial")
			}
			cmd.Process.Signal(tt.sig)
			if code := waitExit(t, cmd); code != tt.code {
				t.Fatalf("exit code %d, want %d; output %q", code, tt.code, out.String())
			}
			if tt.generate {
				select {
				case <-closed:
				case <-time.After(10 * time.Second):
					t.Fatal("the connection was not closed")
				}
			}
			if after := termios(t, slave); after != before {
				t.Error("terminal settings not restored")
			}
			s := out.String()
			if i := strings.LastIndex(s, leaveAltScreen); i < 0 || strings.Contains(s[i:], "YOU >") {
				t.Errorf("alternate screen not left, or an uncommitted exchange printed: %q", s[max(0, i):])
			}
			if files, _ := filepath.Glob(filepath.Join(conversations, "*.json")); len(files) != 0 {
				t.Errorf("saved %v", files)
			}
		})
	}
}

// Outside the fullscreen UI nothing needs restoring, so SIGTERM keeps its
// default effect and is not swallowed.
func TestPlainChatSIGTERMOnPTY(t *testing.T) {
	cmd, _, out, before, _ := startFullscreen(t, nil, "--plain")
	slave := cmd.Stdin.(*os.File)
	out.waitFor(t, "YOU > ")
	cmd.Process.Signal(syscall.SIGTERM)
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) || !exitErr.Sys().(syscall.WaitStatus).Signaled() ||
			exitErr.Sys().(syscall.WaitStatus).Signal() != syscall.SIGTERM {
			t.Errorf("not killed by SIGTERM: %v", err)
		}
	case <-time.After(10 * time.Second):
		cmd.Process.Kill()
		t.Fatal("SIGTERM did not end the plain chat")
	}
	if after := termios(t, slave); after != before {
		t.Error("terminal settings changed")
	}
}

// A real hangup: the terminal goes away during a request. The kernel
// sends SIGHUP; the process must not hang on the dead terminal, and the
// cancelled answer is not saved.
func TestFullscreenHangupOnPTY(t *testing.T) {
	closed := make(chan struct{})
	chat := func(w http.ResponseWriter, r *http.Request) {
		ollamaChunk(w, "Partial")
		<-r.Context().Done()
		close(closed)
	}
	cmd, master, out, _, conversations := startFullscreen(t, chat, "--stream")
	out.waitFor(t, "LLM SETTINGS")
	io.WriteString(master, "Hi\r")
	out.waitFor(t, "Partial")
	master.Close()
	cmd.Stdin.(*os.File).Close() // our copy of the terminal, so it really hangs up
	// 129 when SIGHUP is handled first, 1 when the closed terminal is noticed first.
	code := waitExit(t, cmd)
	if code != 129 && code != 1 {
		t.Errorf("exit code %d", code)
	}
	t.Logf("exit code %d", code)
	select {
	case <-closed:
	case <-time.After(10 * time.Second):
		t.Fatal("the connection was not closed")
	}
	if files, _ := filepath.Glob(filepath.Join(conversations, "*.json")); len(files) != 0 {
		t.Errorf("saved %v", files)
	}
}
