// Package llmtest has test helpers for llm.StreamingProvider
// implementations: a stream collector that checks the event contract, and
// an HTTP transport that tracks whether response bodies are closed.
package llmtest

import (
	"io"
	"iter"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// Result is what a stream delivered.
type Result struct {
	Deltas    []string // answer pieces, in order
	Reasoning []string // reasoning pieces, in order
	Terminal  llm.StreamEvent
}

// Text returns the concatenated answer deltas.
func (r Result) Text() string { return strings.Join(r.Deltas, "") }

// Collect consumes a whole stream and fails the test if it breaks the
// contract: empty deltas, no terminal event, or anything after it, or a
// terminal text that differs from the deltas.
func Collect(t testing.TB, events iter.Seq[llm.StreamEvent]) Result {
	t.Helper()
	var r Result
	for event := range events {
		if r.Terminal != nil {
			t.Errorf("event %#v after the terminal event %#v", event, r.Terminal)
			continue
		}
		switch e := event.(type) {
		case llm.TextDelta:
			if e.Text == "" {
				t.Error("empty TextDelta")
			}
			r.Deltas = append(r.Deltas, e.Text)
		case llm.ReasoningDelta:
			if e.Text == "" {
				t.Error("empty ReasoningDelta")
			}
			r.Reasoning = append(r.Reasoning, e.Text)
		case llm.StreamCompleted, llm.StreamIncomplete, llm.StreamFailed:
			r.Terminal = event
		default:
			t.Errorf("unknown event %#v", event)
		}
	}
	switch e := r.Terminal.(type) {
	case nil:
		t.Error("the stream ended without a terminal event")
	case llm.StreamCompleted:
		if len(r.Deltas) > 0 && e.Response.Text != r.Text() {
			t.Errorf("completed text %q, deltas %q", e.Response.Text, r.Text())
		}
		if e.Response.IncompleteReason != "" {
			t.Errorf("completed with incomplete reason %q", e.Response.IncompleteReason)
		}
	case llm.StreamIncomplete:
		if e.PartialText != r.Text() {
			t.Errorf("partial text %q, deltas %q", e.PartialText, r.Text())
		}
	case llm.StreamFailed:
		if e.PartialText != r.Text() {
			t.Errorf("partial text %q, deltas %q", e.PartialText, r.Text())
		}
		if e.Err == nil {
			t.Error("StreamFailed without an error")
		}
	}
	return r
}

// Transport wraps an http.RoundTripper and records whether every response
// body it returned was closed.
type Transport struct {
	Base   http.RoundTripper
	mu     sync.Mutex
	opened int
	closed int
}

// RoundTrip implements http.RoundTripper.
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.Base.RoundTrip(req)
	if err != nil {
		return resp, err
	}
	t.mu.Lock()
	t.opened++
	t.mu.Unlock()
	resp.Body = &trackedBody{ReadCloser: resp.Body, t: t}
	return resp, nil
}

// Open returns the number of response bodies not closed yet.
func (t *Transport) Open() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.opened - t.closed
}

// Responses returns the number of responses received.
func (t *Transport) Responses() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.opened
}

type trackedBody struct {
	io.ReadCloser
	t    *Transport
	once sync.Once
}

func (b *trackedBody) Close() error {
	b.once.Do(func() {
		b.t.mu.Lock()
		b.t.closed++
		b.t.mu.Unlock()
	})
	return b.ReadCloser.Close()
}
