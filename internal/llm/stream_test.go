package llm_test

import (
	"context"
	"errors"
	"iter"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/llm/llmtest"
)

// scripted is a StreamingProvider that replays events; it needs no
// network, SDK or terminal.
type scripted struct {
	llm.Provider // unused methods
	events       []llm.StreamEvent
	stopped      atomic.Bool // the consumer stopped early
}

func (s *scripted) Stream(ctx context.Context, req llm.Request) iter.Seq[llm.StreamEvent] {
	return func(yield func(llm.StreamEvent) bool) {
		for _, e := range s.events {
			if ctx.Err() != nil {
				yield(llm.StreamFailed{Err: ctx.Err()})
				return
			}
			if !yield(e) {
				s.stopped.Store(true)
				return
			}
		}
	}
}

// outcome is what a chat UI (Task 004B) needs from a stream: the text it
// showed, and whether to commit it.
type outcome struct {
	shown  string
	commit string // the answer to save; "" if nothing is committed
	notice string
}

// consume is how a chat consumes a stream, independently of os.Stdout:
// show deltas, commit only a completed answer.
func consume(events iter.Seq[llm.StreamEvent]) outcome {
	var o outcome
	var shown strings.Builder
	for event := range events {
		switch e := event.(type) {
		case llm.TextDelta:
			shown.WriteString(e.Text)
		case llm.ReasoningDelta:
			// never shown
		case llm.StreamCompleted:
			o.commit = e.Response.Text
		case llm.StreamIncomplete:
			o.notice = "Response incomplete: " + e.Reason + "."
		case llm.StreamFailed:
			if errors.Is(e.Err, context.Canceled) {
				o.notice = "Request cancelled."
			} else {
				o.notice = "error: " + e.Err.Error()
			}
		}
	}
	o.shown = shown.String()
	return o
}

func TestConsumeTerminalOutcomes(t *testing.T) {
	apiErr := &llm.Error{Kind: llm.KindRateLimit, Message: "Slow down."}
	tests := []struct {
		name   string
		events []llm.StreamEvent
		want   outcome
	}{
		{"completed", []llm.StreamEvent{llm.ReasoningDelta{Text: "hmm"}, llm.TextDelta{Text: "Hi "}, llm.TextDelta{Text: "there"},
			llm.StreamCompleted{Response: llm.Response{Text: "Hi there"}}}, outcome{shown: "Hi there", commit: "Hi there"}},
		{"incomplete", []llm.StreamEvent{llm.TextDelta{Text: "Hi"}, llm.StreamIncomplete{Reason: "the max output token limit was reached", PartialText: "Hi"}},
			outcome{shown: "Hi", notice: "Response incomplete: the max output token limit was reached."}},
		{"failed", []llm.StreamEvent{llm.TextDelta{Text: "Hi"}, llm.StreamFailed{Err: apiErr, PartialText: "Hi"}},
			outcome{shown: "Hi", notice: "error: Slow down."}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &scripted{events: tt.events}
			llmtest.Collect(t, p.Stream(context.Background(), llm.Request{}))
			if got := consume(p.Stream(context.Background(), llm.Request{})); got != tt.want {
				t.Errorf("got %+v, want %+v", got, tt.want)
			}
		})
	}
	if llm.ExitCode(apiErr) != 5 { // a failure keeps its kind, and so its exit code
		t.Errorf("exit code %d", llm.ExitCode(apiErr))
	}
}

func TestConsumeCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := &scripted{events: []llm.StreamEvent{llm.TextDelta{Text: "a"}, llm.TextDelta{Text: "b"},
		llm.StreamCompleted{Response: llm.Response{Text: "ab"}}}}
	var o outcome
	events := func(yield func(llm.StreamEvent) bool) {
		for e := range p.Stream(ctx, llm.Request{}) {
			if !yield(e) {
				return
			}
			cancel() // Ctrl+C after the first event
		}
	}
	o = consume(events)
	if o != (outcome{shown: "a", notice: "Request cancelled."}) {
		t.Errorf("got %+v", o)
	}

	// A consumer that stops early stops the producer.
	p = &scripted{events: []llm.StreamEvent{llm.TextDelta{Text: "a"}, llm.TextDelta{Text: "b"}}}
	for range p.Stream(context.Background(), llm.Request{}) {
		break
	}
	if !p.stopped.Load() {
		t.Error("the producer did not notice the consumer stopped")
	}
}

func TestIdleTimer(t *testing.T) {
	fired := make(chan struct{}, 1)
	timer := llm.NewIdleTimer(20*time.Millisecond, func() { fired <- struct{}{} })
	timer.Stop()
	time.Sleep(40 * time.Millisecond) // stopped: does not fire
	select {
	case <-fired:
		t.Fatal("fired while stopped")
	default:
	}
	timer.Reset()
	select {
	case <-fired:
	case <-time.After(5 * time.Second):
		t.Fatal("did not fire")
	}
	timer.Reset() // after expiry: no second call
	time.Sleep(40 * time.Millisecond)
	select {
	case <-fired:
		t.Error("fired twice")
	default:
	}
}
