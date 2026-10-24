package tui

import (
	"context"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// TestExitRestoresTheTerminal checks every exit path: the screen is
// finalized once, and the transcript (committed messages only) is
// printed after that.
func TestExitRestoresTheTerminal(t *testing.T) {
	tests := []struct {
		name string
		exit func(h *harness)
		code int
	}{
		{"/exit", func(h *harness) { h.submit("/exit") }, ExitOK},
		{"Ctrl+D", func(h *harness) { h.key(tcell.KeyCtrlD) }, ExitOK},
		{"Ctrl+C", func(h *harness) { h.key(tcell.KeyCtrlC) }, ExitInterrupted},
		{"canceled context", func(h *harness) { h.cancel() }, ExitInterrupted},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeProvider{turns: []turnScript{
				events(completed("First answer\n\n  indented", 1, 1)),
				events(append(deltas("Partial"), llm.StreamFailed{Err: &llm.Error{Kind: llm.KindAPI, Message: "boom"}})...),
				events(completed("Second answer", 1, 1)),
			}}
			h := start(t, fake, setup{streamed: true, transcript: true})
			h.submit("Q1")
			h.waitText("AI > First answer")
			h.submit("Failing question")
			h.waitText("Error: boom")
			h.submit("Q3")
			h.waitText("AI > Second answer")
			tt.exit(h)
			if code := h.exitCode(); code != tt.code {
				t.Errorf("code %d, want %d", code, tt.code)
			}
			if _, finis := h.screen.counts(); finis != 1 {
				t.Errorf("Fini called %d times", finis)
			}
			want := "YOU > Q1\n\nAI > First answer\n\n  indented\n\nYOU > Q3\n\nAI > Second answer\n"
			if got := h.stdout.String(); got != want {
				t.Errorf("transcript %q, want %q", got, want)
			}
			if h.log[0] != "fini" || h.stderr.String() != "" {
				t.Errorf("order %q stderr %q", h.log, h.stderr.String())
			}
		})
	}
}

func TestNoTranscriptForAnEmptyConversation(t *testing.T) {
	h := start(t, &fakeProvider{}, setup{transcript: true})
	h.key(tcell.KeyCtrlD)
	if code := h.exitCode(); code != ExitOK || h.stdout.String() != "" {
		t.Errorf("code %d stdout %q", code, h.stdout.String())
	}
	h = start(t, &fakeProvider{turns: []turnScript{events(completed("A", 1, 1))}}, setup{})
	h.submit("Q")
	h.waitText("AI > A")
	h.key(tcell.KeyCtrlD)
	if h.exitCode(); h.stdout.String() != "" {
		t.Errorf("transcript without --transcript-on-exit: %q", h.stdout.String())
	}
}

func TestPanicRestoresTheTerminal(t *testing.T) {
	fake := &fakeProvider{turns: []turnScript{events(completed("Kept", 1, 1))}}
	h := start(t, fake, setup{streamed: true, transcript: true})
	h.submit("Q")
	h.waitText("AI > Kept")
	h.screen.armPanic()
	h.key(tcell.KeyLeft)
	if code := h.exitCode(); code != ExitError {
		t.Fatalf("code %d", code)
	}
	if _, finis := h.screen.counts(); finis != 1 {
		t.Errorf("Fini called %d times", finis)
	}
	if got := h.stderr.String(); got != "neurobrau: error: the fullscreen UI failed: simulated render failure\n" {
		t.Errorf("stderr %q", got)
	}
	if h.log[0] != "fini" || h.stdout.String() != "YOU > Q\n\nAI > Kept\n" {
		t.Errorf("order %q, transcript %q", h.log, h.stdout.String())
	}
	if strings.Contains(h.stderr.String(), "goroutine") {
		t.Error("stack trace printed")
	}
}

// The session is never locked while a request runs: a provider (here the
// fake) can use it during the request, and the UI keeps drawing.
func TestNoSessionLockDuringRequest(t *testing.T) {
	var h *harness
	gate := make(chan struct{})
	fake := &fakeProvider{turns: []turnScript{func(ctx context.Context, yield func(llm.StreamEvent) bool) {
		h.sess.Messages() // would deadlock if the UI held the session lock
		h.sess.Requests()
		yield(llm.TextDelta{Text: "ok"})
		<-gate
		yield(completed("ok", 1, 1))
	}}}
	h = start(t, fake, setup{streamed: true})
	h.submit("Q")
	h.waitText("AI > ok")
	close(gate)
	h.waitStatus("DONE")
}

func TestNoGoroutinesLeft(t *testing.T) {
	before := runtime.NumGoroutine()
	fake := &fakeProvider{turns: []turnScript{
		events(completed("A", 1, 1)),
		func(ctx context.Context, yield func(llm.StreamEvent) bool) {
			for yield(llm.TextDelta{Text: "x"}) {
			}
		},
		gated(make(chan struct{}), deltas("open")),
	}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1")
	h.waitText("AI > A")
	h.submit("Q2")
	h.waitText("AI > xx")
	h.key(tcell.KeyCtrlC)
	h.waitStatus("CANCELLED")
	h.submit("Q3")
	h.waitText("AI > open")
	h.key(tcell.KeyCtrlD) // ignored while generating
	h.submit("/exit")     // cancels the open request
	if code := h.exitCode(); code != ExitOK {
		t.Fatalf("code %d", code)
	}
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > before {
		if time.Now().After(deadline) {
			buf := make([]byte, 1<<16)
			t.Fatalf("%d goroutines, %d before:\n%s", runtime.NumGoroutine(), before, buf[:runtime.Stack(buf, true)])
		}
		time.Sleep(5 * time.Millisecond)
	}
	if _, finished := fake.state(); finished != 3 {
		t.Errorf("%d streams finished", finished)
	}
	if sent, _ := fake.state(); !reflect.DeepEqual(len(sent), 3) {
		t.Errorf("sent %d", len(sent))
	}
}
