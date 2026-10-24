package tui

import (
	"context"
	"fmt"
)

// operation is a provider check or discovery running in the background
// (F2–F5), e.g. "Loading Ollama models". Its work runs in a goroutine
// that never touches the UI: it returns a function that the event loop
// calls to show the result. The result channel is the operation's
// generation token: once the operation is retired (canceled, or the
// result arrived) nothing else from it is applied.
type operation struct {
	label  string // status detail: CONNECTING | <label>
	cancel context.CancelFunc
	result chan func()   // capacity 1, so the goroutine never blocks
	done   chan struct{} // closed when the goroutine has finished
}

// startOp runs work in the background with a cancelable context. Only one
// background job (request or operation) runs at a time, because providers
// are not safe for concurrent use; callers check idle first.
func (a *app) startOp(label string, work func(ctx context.Context) func()) {
	ctx, cancel := context.WithCancel(a.ctx)
	op := &operation{label: label, cancel: cancel, result: make(chan func(), 1), done: make(chan struct{})}
	go func() {
		defer close(op.done)
		defer func() {
			if r := recover(); r != nil { // a provider bug must not end the process with the terminal raw
				op.result <- func() { a.showError(fmt.Sprintf("Unexpected error: %v", r)) }
			}
		}()
		op.result <- work(ctx)
	}()
	a.op = op
}

// finishOp applies the result of the running operation.
func (a *app) finishOp(apply func()) {
	a.retireOp()
	a.transcript.scroll = 0
	apply()
}

// retireOp stops listening to the running operation; the loop waits for
// its goroutine to finish before the provider is used again.
func (a *app) retireOp() {
	a.op.cancel()
	a.opClosing = a.op.done
	a.op = nil
}

// cancelOp is Ctrl+C during a check or discovery.
func (a *app) cancelOp() {
	a.retireOp()
	a.addEntry(entryError, "Cancelled.")
	a.status = status{state: stateCancelled}
}

// busy reports a running request or operation.
func (a *app) busy() bool { return a.active != nil || a.op != nil }

// idle waits briefly for canceled background work to finish; false (with
// a warning) if it is still closing. Starting new work before that would
// use a provider from two goroutines at once.
func (a *app) idle() bool {
	if (a.closing != nil || a.opClosing != nil) && !a.waitClosed(closeWait) {
		a.status = warning("The cancelled response is still closing; try again shortly")
		return false
	}
	return true
}
