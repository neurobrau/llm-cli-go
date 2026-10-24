package tui

import (
	"context"
	"fmt"
	"iter"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// send starts a request for text in a worker goroutine
// (docs/tui-spec.md, sections 8 and 14). The worker hands its events to
// the event loop through llm.StreamChannel's bounded channel; it never
// draws or touches the session. The context is a snapshot of the
// committed messages, so no lock is held while the request runs.
func (a *app) send(text string) {
	if !a.idle() {
		// Only one request at a time: the previous one has not closed yet.
		a.input.set(text)
		return
	}
	a.addEntry(entryUser, text)
	messages := a.Session.Context(text)
	ctx, cancel := context.WithCancel(a.ctx)
	// The request is built here, so later F2–F6 changes cannot affect it.
	var events iter.Seq[llm.StreamEvent]
	if a.Stream {
		events = a.Engine.Stream(ctx, messages)
	} else {
		p, req := a.Engine.Provider, a.Engine.Request(messages)
		events = replyEvents(ctx, func(ctx context.Context) (llm.Response, error) { return p.Chat(ctx, req) })
	}
	a.active = &request{
		text:     text,
		streamed: a.Stream,
		cancel:   cancel,
		events:   llm.StreamChannel(ctx, recovered(events)),
		started:  a.now(),
	}
}

// replyEvents runs a non-streaming request as a stream with only its
// terminal event, so both kinds of request share the worker and the
// event handling.
func replyEvents(ctx context.Context, reply func(context.Context) (llm.Response, error)) iter.Seq[llm.StreamEvent] {
	return func(yield func(llm.StreamEvent) bool) {
		resp, err := reply(ctx)
		switch {
		case err != nil:
			yield(llm.StreamFailed{Err: err})
		case resp.IncompleteReason != "":
			yield(llm.StreamIncomplete{Reason: resp.IncompleteReason, PartialText: resp.Text,
				Usage: resp.Usage, ServiceTier: resp.ServiceTier})
		default:
			yield(llm.StreamCompleted{Response: resp})
		}
	}
}

// recovered turns a panic in a provider into StreamFailed: a panic in a
// worker goroutine would end the process without restoring the terminal.
func recovered(events iter.Seq[llm.StreamEvent]) iter.Seq[llm.StreamEvent] {
	return func(yield func(llm.StreamEvent) bool) {
		stopped := false
		defer func() {
			if r := recover(); r != nil && !stopped {
				yield(llm.StreamFailed{Err: &llm.Error{Kind: llm.KindAPI,
					Message: fmt.Sprintf("Unexpected error while streaming: %v", r)}})
			}
		}()
		for event := range events {
			if !yield(event) {
				stopped = true
				return
			}
		}
	}
}

// waitClosed waits at most d for the workers of retired requests and
// operations to finish; remaining events are dropped.
func (a *app) waitClosed(d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	for a.closing != nil || a.opClosing != nil {
		select {
		case _, ok := <-a.closing:
			if !ok {
				a.closing = nil
			}
		case <-a.opClosing:
			a.opClosing = nil
		case <-timer.C:
			return false
		}
	}
	return true
}

// retire ends the active request: later events are dropped, and the
// event loop drains its channel until the worker is done.
func (a *app) retire() *request {
	req := a.active
	a.active = nil
	req.cancel()
	a.closing = req.events
	return req
}

// apply handles one event of the active request; ok is false when its
// channel closed without a terminal event.
func (a *app) apply(event llm.StreamEvent, ok bool) {
	req := a.active
	if !ok {
		a.retire()
		a.failed(req, "The response stream ended without a result.")
		return
	}
	switch e := event.(type) {
	case llm.ReasoningDelta:
		req.thinking = true // shown in the status only, never as text
	case llm.TextDelta:
		if req.firstText.IsZero() {
			req.firstText = a.now()
		}
		a.showAnswer(req, req.answer+e.Text)
	case llm.StreamCompleted:
		a.retire()
		a.completed(req, e.Response)
	case llm.StreamIncomplete:
		a.retire()
		if !req.streamed && e.PartialText != "" {
			a.showAnswer(req, e.PartialText) // shown, but not part of the conversation
		}
		a.addEntry(entryError, fmt.Sprintf("Incomplete: %s.%s", e.Reason, notSaved(req)))
		a.status = warning("Response incomplete: " + e.Reason)
		// Billed, so its reported usage is kept (session semantics).
		if err := a.Session.RecordIncomplete(llm.Response{Text: e.PartialText, IncompleteReason: e.Reason,
			Usage: e.Usage, ServiceTier: e.ServiceTier}); err != nil {
			a.addEntry(entryError, "Warning: "+err.Error())
		}
	case llm.StreamFailed:
		a.retire()
		a.failed(req, e.Err.Error())
	}
}

// completed commits and saves the exchange, once, and shows the final
// text and the metrics.
func (a *app) completed(req *request, resp llm.Response) {
	a.showAnswer(req, resp.Text) // an empty answer still gets its entry
	saveErr := a.Session.Commit(req.text, resp)
	var firstText time.Duration
	if req.streamed && !req.firstText.IsZero() {
		firstText = req.firstText.Sub(req.started)
	}
	metrics := resp
	if !req.streamed {
		metrics.Timing = llm.Timing{} // a complete answer shows the observed latency only
	}
	a.status = doneStatus(metrics, a.now().Sub(req.started), firstText)
	switch {
	case saveErr != nil:
		a.addEntry(entryError, "Warning: "+saveErr.Error())
		s := warning("Conversation not saved")
		a.afterTransient = &s
	case a.Session.Persistent():
		s := ready("Conversation saved")
		a.afterTransient = &s
	}
}

func (a *app) failed(req *request, msg string) {
	a.addEntry(entryError, "Error: "+msg+notSaved(req))
	a.status = status{stateError, []string{msg}}
}

// showAnswer sets the text of the request's AI entry, creating it first.
func (a *app) showAnswer(req *request, text string) {
	req.answer = text
	a.anchored(func() {
		if req.entry == nil {
			req.entry = &entry{kind: entryAI}
			a.transcript.entries = append(a.transcript.entries, req.entry)
		}
		req.entry.setText(text)
	})
}

// cancelActive is Ctrl+C while a request runs: its id is retired at once
// and the partial text stays visible but is never committed. The worker
// stops and closes the stream in the background.
func (a *app) cancelActive() {
	if a.active == nil {
		return
	}
	req := a.retire()
	if !req.streamed {
		a.addEntry(entryError, "Request cancelled.")
		a.status = status{state: stateCancelled}
		return
	}
	a.addEntry(entryError, "Cancelled."+notSaved(req))
	a.status = status{state: stateCancelled}
	if req.answer != "" {
		a.status.parts = []string{"Partial response not saved"}
	}
}

// stopRequests cancels the running request on exit and waits a little
// for the workers to close their streams; a worker that is still blocked
// never delays the exit.
func (a *app) stopRequests() {
	a.cancelActive()
	if a.op != nil {
		a.retireOp()
	}
	a.waitClosed(exitWait)
}

func notSaved(req *request) string {
	if req.answer != "" {
		return " The partial answer above was not saved."
	}
	return ""
}
