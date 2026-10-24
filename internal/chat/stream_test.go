package chat

import (
	"context"
	"iter"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// turnScript produces the events of one streamed request. It must stop
// when yield returns false and should watch ctx like a real provider.
type turnScript func(ctx context.Context, yield func(llm.StreamEvent) bool)

// fakeStream replays one script per request and records what was sent
// and whether each stream was finished (its "connection" closed).
type fakeStream struct {
	mu       sync.Mutex
	turns    []turnScript
	sent     [][]llm.Message
	finished int
}

func (f *fakeStream) Stream(ctx context.Context, messages []llm.Message) iter.Seq[llm.StreamEvent] {
	return func(yield func(llm.StreamEvent) bool) {
		f.mu.Lock()
		f.sent = append(f.sent, messages)
		script := f.turns[0]
		f.turns = f.turns[1:]
		f.mu.Unlock()
		defer func() {
			f.mu.Lock()
			f.finished++
			f.mu.Unlock()
		}()
		script(ctx, yield)
	}
}

func (f *fakeStream) state() ([][]llm.Message, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.sent, f.finished
}

// events yields the events in order.
func events(evs ...llm.StreamEvent) turnScript {
	return func(ctx context.Context, yield func(llm.StreamEvent) bool) {
		for _, e := range evs {
			if !yield(e) {
				return
			}
		}
	}
}

// stall yields evs, then waits for cancellation like a stalled server.
func stall(evs ...llm.StreamEvent) turnScript {
	return func(ctx context.Context, yield func(llm.StreamEvent) bool) {
		for _, e := range evs {
			if !yield(e) {
				return
			}
		}
		<-ctx.Done()
		yield(llm.StreamFailed{Err: ctx.Err()})
	}
}

func completedWith(text string, input, output int) llm.StreamCompleted {
	return llm.StreamCompleted{Response: llm.Response{Text: text, ServiceTier: "default",
		Usage: &llm.Usage{InputTokens: intp(input), OutputTokens: intp(output), TotalTokens: intp(input + output)}}}
}

func deltas(parts ...string) []llm.StreamEvent {
	var evs []llm.StreamEvent
	for _, p := range parts {
		evs = append(evs, llm.TextDelta{Text: p})
	}
	return evs
}

func streamHarness(t *testing.T, history bool, turns ...turnScript) *harness {
	h := newHarness(t, history)
	h.stream = &fakeStream{turns: turns}
	return h
}

func (h *harness) saved(t *testing.T) storage.Conversation {
	t.Helper()
	c, err := h.store.Load(h.store.PathFor(h.session.Conversation().ID))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestStreamTurns(t *testing.T) {
	h := streamHarness(t, true,
		events(append(deltas("Hel", "lo!\nSecond line"), completedWith("Hello!\nSecond line", 10, 3))...),
		events(llm.ReasoningDelta{Text: "SECRET THINKING"}, llm.TextDelta{Text: "Fine."},
			llm.StreamCompleted{Response: llm.Response{Text: "Fine.", Reasoning: "SECRET THINKING",
				Usage:  &llm.Usage{InputTokens: intp(25), OutputTokens: intp(9), ReasoningTokens: intp(7)},
				Timing: llm.Timing{Total: 2 * time.Second, Eval: time.Second}}}),
	)
	h.showUsage = true
	if code := h.run(context.Background(), "Hi\nHow are you?\n"); code != ExitOK {
		t.Fatalf("exit %d", code)
	}
	want := banner + "YOU > AI > Hello!\nSecond line\nYOU > AI > Fine.\nYOU > \nGoodbye.\n"
	if h.stdout.String() != want {
		t.Errorf("stdout %q\nwant   %q", h.stdout.String(), want)
	}
	if h.stderr.String() != "Token usage: input=10, output=3, total=13\nToken usage: input=25, output=9\nDuration: 2.00 s (generation 1.00 s)\n" {
		t.Errorf("stderr %q", h.stderr.String())
	}
	sent, finished := h.stream.state()
	if !reflect.DeepEqual(sent, [][]llm.Message{msgs("Hi"), msgs("Hi", "Hello!\nSecond line", "How are you?")}) || finished != 2 {
		t.Errorf("sent %q, %d finished", sent, finished)
	}
	c := h.saved(t)
	if !reflect.DeepEqual(c.Messages, msgs("Hi", "Hello!\nSecond line", "How are you?", "Fine.")) || len(c.Requests) != 2 ||
		c.Requests[1].Status != storage.StatusCompleted || *c.Requests[1].Usage.ReasoningTokens != 7 {
		t.Errorf("saved %+v", c)
	}
	for _, m := range c.Messages {
		if strings.Contains(m.Content, "SECRET") || strings.Contains(m.Content, AIPrefix) {
			t.Errorf("saved message %q", m.Content)
		}
	}
	if strings.Contains(h.stdout.String(), "SECRET") {
		t.Error("reasoning was printed")
	}
}

func TestStreamEmptyAndReasoningOnlyAnswers(t *testing.T) {
	h := streamHarness(t, true,
		events(llm.StreamCompleted{Response: llm.Response{}}),
		events(llm.ReasoningDelta{Text: "only thinking"}, llm.StreamCompleted{Response: llm.Response{Reasoning: "only thinking"}}),
	)
	h.run(context.Background(), "Q1\nQ2\n")
	if want := banner + "YOU > AI > \nYOU > AI > \nYOU > \nGoodbye.\n"; h.stdout.String() != want {
		t.Errorf("stdout %q", h.stdout.String())
	}
	if c := h.saved(t); !reflect.DeepEqual(c.Messages, msgs("Q1", "", "Q2", "")) || c.Requests != nil {
		t.Errorf("saved %+v", c)
	}
}

// TestStreamOutputIsIncremental checks that a delta is on stdout before
// the stream goes on: the script waits until it sees it there.
func TestStreamOutputIsIncremental(t *testing.T) {
	var h *harness
	h = streamHarness(t, false, func(ctx context.Context, yield func(llm.StreamEvent) bool) {
		yield(llm.TextDelta{Text: "First"})
		waitFor(t, h.stdout, "AI > First")
		yield(llm.TextDelta{Text: " second"})
		waitFor(t, h.stdout, "AI > First second")
		yield(completedWith("First second", 1, 2))
	})
	h.run(context.Background(), "Q\n")
	if !strings.Contains(h.stdout.String(), "YOU > AI > First second\nYOU > ") || strings.Count(h.stdout.String(), AIPrefix) != 1 {
		t.Errorf("stdout %q", h.stdout.String())
	}
}

func TestStreamIncompleteAndFailed(t *testing.T) {
	h := streamHarness(t, true,
		events(llm.TextDelta{Text: "A1"}, completedWith("A1", 5, 1)),
		events(llm.TextDelta{Text: "Part"}, llm.StreamIncomplete{Reason: "the max output token limit was reached", PartialText: "Part",
			Usage: &llm.Usage{InputTokens: intp(7), OutputTokens: intp(50)}, ServiceTier: "flex"}),
		events(llm.StreamIncomplete{Reason: "the content filter stopped the response"}), // no text, no usage
		events(llm.TextDelta{Text: "Broken"}, llm.StreamFailed{Err: &llm.Error{Kind: llm.KindConnection, Message: "Connection lost."}, PartialText: "Broken"}),
		events(llm.StreamFailed{Err: &llm.Error{Kind: llm.KindRateLimit, Message: "Slow down."}}),
		events(llm.TextDelta{Text: "A6"}, completedWith("A6", 9, 1)),
	)
	h.run(context.Background(), "Q1\nQ2\nQ3\nQ4\nQ5\nQ6\n/usage\n")
	out := h.stdout.String()
	for _, want := range []string{"YOU > AI > Part\nYOU > YOU > AI > Broken\nYOU > YOU > AI > A6\n",
		"Conversation usage (3 request(s), as reported by the API):\nInput tokens: 21\nOutput tokens: 52\nTotal tokens: 16\n",
		"Latest request input: 9 tokens", "Includes 1 incomplete answer(s)"} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout lacks %q:\n%s", want, out)
		}
	}
	wantErr := "warning: Response incomplete: the max output token limit was reached. The partial response was not saved.\n" +
		"warning: Response incomplete: the content filter stopped the response.\n" +
		"error: Connection lost. The partial response was not saved.\n" +
		"error: Slow down.\n"
	if h.stderr.String() != wantErr {
		t.Errorf("stderr %q", h.stderr.String())
	}
	sent, _ := h.stream.state()
	if !reflect.DeepEqual(sent[5], msgs("Q1", "A1", "Q6")) {
		t.Errorf("last request %q", sent[5])
	}
	c := h.saved(t)
	if !reflect.DeepEqual(c.Messages, msgs("Q1", "A1", "Q6", "A6")) || len(c.Requests) != 3 ||
		c.Requests[1].Status != storage.StatusIncomplete || c.Requests[1].ServiceTier != "flex" {
		t.Errorf("saved %+v", c)
	}
}

func TestStreamEventsAfterTerminalAreDropped(t *testing.T) {
	h := streamHarness(t, true, events(llm.TextDelta{Text: "A"}, completedWith("A", 1, 1),
		llm.TextDelta{Text: "LATE"}, completedWith("ALATE", 1, 1)))
	h.run(context.Background(), "Q\n")
	if strings.Contains(h.stdout.String(), "LATE") {
		t.Errorf("stdout %q", h.stdout.String())
	}
	if c := h.saved(t); len(c.Messages) != 2 || len(c.Requests) != 1 {
		t.Errorf("saved %+v", c)
	}
}

func TestStreamCtrlC(t *testing.T) {
	for _, tt := range []struct {
		name   string
		before []llm.StreamEvent
		stdout string
		stderr string
	}{
		{"after text", []llm.StreamEvent{llm.ReasoningDelta{Text: "x"}, llm.TextDelta{Text: "Partial"}}, "YOU > AI > Partial\nYOU > AI > A2\n",
			"\nRequest cancelled. The partial response was not saved.\n"},
		{"before the first delta", nil, "YOU > YOU > AI > A2\n", "\nRequest cancelled.\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var h *harness
			script := func(ctx context.Context, yield func(llm.StreamEvent) bool) {
				for _, e := range tt.before {
					yield(e)
				}
				if len(tt.before) > 0 {
					waitFor(t, h.stdout, "AI > Partial")
				}
				h.interrupts <- os.Interrupt // Ctrl+C while the server stalls
				stall()(ctx, yield)
			}
			h = streamHarness(t, true, script, events(llm.TextDelta{Text: "A2"}, completedWith("A2", 1, 1)))
			if code := h.run(context.Background(), "Q1\nQ2\n"); code != ExitOK {
				t.Errorf("exit %d", code)
			}
			if !strings.Contains(h.stdout.String(), tt.stdout) || h.stderr.String() != tt.stderr {
				t.Errorf("stdout %q stderr %q", h.stdout.String(), h.stderr.String())
			}
			sent, finished := h.stream.state()
			if !reflect.DeepEqual(sent[1], msgs("Q2")) || finished != 2 {
				t.Errorf("second request %q, %d finished", sent[1], finished)
			}
			if c := h.saved(t); !reflect.DeepEqual(c.Messages, msgs("Q2", "A2")) {
				t.Errorf("saved %q", c.Messages)
			}
		})
	}
}

func TestStreamCanceledContextEndsChat(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	h := streamHarness(t, true, func(sctx context.Context, yield func(llm.StreamEvent) bool) {
		yield(llm.TextDelta{Text: "Part"})
		cancel()
		stall()(sctx, yield)
	})
	if code := h.run(ctx, "Q\nnever\n"); code != ExitInterrupted || len(h.session.Messages()) != 0 {
		t.Errorf("code %d", code)
	}
	if n, _ := h.store.CountConversations(); n != 0 {
		t.Error("canceled answer saved")
	}
}

// pipeWriter fails like a pipe whose reader went away.
type pipeWriter struct {
	mu     sync.Mutex
	writes int
}

func (w *pipeWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.writes++
	return 0, syscall.EPIPE
}

func TestStreamBrokenStdout(t *testing.T) {
	h := streamHarness(t, true, stall(llm.TextDelta{Text: "A"}))
	out := &pipeWriter{}
	engine := &Engine{Provider: h.provider}
	code := Run(context.Background(), Config{Session: h.session, Reply: engine.Reply, Stream: h.stream.Stream,
		Stdin: strings.NewReader("Q\n"), Stdout: out, Stderr: h.stderr})
	_, finished := h.stream.state()
	if code != ExitBrokenPipe || finished != 1 || h.stderr.String() != "" || len(h.session.Messages()) != 0 {
		t.Errorf("code %d, %d finished, stderr %q", code, finished, h.stderr.String())
	}
}

func TestStreamSaveFailureIsAWarning(t *testing.T) {
	h := streamHarness(t, true, events(llm.TextDelta{Text: "A1"}, completedWith("A1", 1, 1)),
		events(llm.TextDelta{Text: "A2"}, completedWith("A2", 1, 1)))
	writeFileForTest(t, h.store.Dir(), "a file") // the conversations path is a file
	h.run(context.Background(), "Q1\nQ2\n")
	if strings.Count(h.stderr.String(), "warning: Could not save conversation history at ") != 2 {
		t.Errorf("stderr %q", h.stderr.String())
	}
	if sent, _ := h.stream.state(); !reflect.DeepEqual(sent[1], msgs("Q1", "A1", "Q2")) {
		t.Errorf("second request %q", sent[1])
	}
}

func TestStreamCommandsWithMixedOutcomes(t *testing.T) {
	h := streamHarness(t, true,
		events(llm.TextDelta{Text: "A1"}, completedWith("A1", 3, 1)),
		events(llm.StreamIncomplete{Reason: "r", Usage: &llm.Usage{InputTokens: intp(4)}}),
		events(llm.TextDelta{Text: "A3"}, completedWith("A3", 5, 1)),
	)
	h.run(context.Background(), "Q1\nQ2\n/reset\n/usage\nQ3\n/cleanup\n  Yes \n/usage\n")
	out := h.stdout.String()
	for _, want := range []string{
		"Started a new conversation. The previous one stays saved.\nYOU > Token usage: no API requests in this conversation yet.\n",
		"Delete 2 saved conversation(s) in " + h.store.Dir(),
		"Deleted 2 saved conversation(s). Started a new conversation.\nYOU > Token usage: no API requests in this conversation yet.\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout lacks %q:\n%s", want, out)
		}
	}
	if n, _ := h.store.CountConversations(); n != 0 {
		t.Errorf("%d files", n)
	}
}

func TestStreamNoHistory(t *testing.T) {
	h := streamHarness(t, false, events(llm.TextDelta{Text: "A1"}, completedWith("A1", 1, 1)),
		events(llm.StreamIncomplete{Reason: "r", Usage: &llm.Usage{InputTokens: intp(1)}}),
		events(llm.TextDelta{Text: "A3"}, completedWith("A3", 1, 1)))
	h.run(context.Background(), "Q1\nQ2\n/reset\nQ3\n")
	if sent, _ := h.stream.state(); !reflect.DeepEqual(sent[2], msgs("Q3")) {
		t.Errorf("after /reset %q", sent[2])
	}
	if h.session.Persistent() || len(h.session.Requests()) != 1 {
		t.Errorf("requests %+v", h.session.Requests())
	}
}

func writeFileForTest(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
