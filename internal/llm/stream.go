package llm

import (
	"context"
	"iter"
	"sync"
	"time"
)

// StreamEvent is one event of a streamed answer (docs/tui-spec.md,
// section 14): any number of TextDelta and ReasoningDelta events, then
// exactly one terminal event, StreamCompleted, StreamIncomplete or
// StreamFailed. Nothing follows the terminal event. Consumers tell the
// events apart with a type switch.
type StreamEvent interface {
	streamEvent()
}

// TextDelta is a non-empty piece of the answer.
type TextDelta struct{ Text string }

// ReasoningDelta is a non-empty piece of the model's thinking. It is never
// part of the answer and must not be shown or saved as such.
type ReasoningDelta struct{ Text string }

// StreamCompleted ends a stream that finished normally. Response.Text is
// the whole answer (the concatenated deltas); Response also holds the
// reported usage and metadata (ID, model, service tier, stop reason,
// timing). IncompleteReason is always empty.
type StreamCompleted struct{ Response Response }

// StreamIncomplete ends a stream that stopped early (token limit, content
// filter). The partial text is not a complete answer; its usage was billed.
type StreamIncomplete struct {
	Reason      string // readable, e.g. "the max output token limit was reached"
	PartialText string
	Usage       *Usage // nil: not reported
	ServiceTier string // as reported, if any
}

// StreamFailed ends a stream whose request or connection failed. Err is an
// *Error, or wraps context.Canceled when the caller canceled the request.
// PartialText is the answer text that arrived before the failure.
type StreamFailed struct {
	Err         error
	PartialText string
}

func (TextDelta) streamEvent()        {}
func (ReasoningDelta) streamEvent()   {}
func (StreamCompleted) streamEvent()  {}
func (StreamIncomplete) streamEvent() {}
func (StreamFailed) streamEvent()     {}

// StreamingProvider is a Provider that can also stream its answers. It is
// separate from Provider so that fakes and callers that never stream need
// not implement it.
type StreamingProvider interface {
	Provider
	// Stream sends a request and returns its events as an iterator. The
	// request is sent when the iteration starts, in the caller's goroutine;
	// no goroutine outlives it. Every return path closes the HTTP response:
	// after the terminal event, when the consumer stops early (break), and
	// when ctx is canceled, which ends the stream promptly with a
	// StreamFailed wrapping context.Canceled. Failures are events, never
	// panics. A stream is meant to be iterated once.
	Stream(ctx context.Context, req Request) iter.Seq[StreamEvent]
}

// IdleTimer cancels a stream that waits too long for data: unlike a
// timeout for the whole request, it never ends a healthy long answer, and
// time the consumer spends handling events does not count. It runs from
// creation; Stop it before handing data to the consumer and Reset it
// before waiting for more.
type IdleTimer struct {
	mu      sync.Mutex
	timer   *time.Timer
	timeout time.Duration
	expired bool
}

// NewIdleTimer calls expire once if timeout passes while it runs.
func NewIdleTimer(timeout time.Duration, expire func()) *IdleTimer {
	t := &IdleTimer{timeout: timeout}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.timer = time.AfterFunc(timeout, func() {
		t.mu.Lock()
		t.expired = true
		t.mu.Unlock()
		expire()
	})
	return t
}

// Reset starts a new full timeout, unless the timer already expired.
func (t *IdleTimer) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.expired {
		t.timer.Reset(t.timeout)
	}
}

// Stop pauses the timer until the next Reset.
func (t *IdleTimer) Stop() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.timer.Stop()
}
