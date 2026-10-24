package chat

import (
	"context"
	"errors"
	"fmt"
	"io"
	"syscall"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// streamTurn is what one streamed request showed and how it ended.
type streamTurn struct {
	printed     bool            // "AI > " and some text were written
	terminal    llm.StreamEvent // the first terminal event, or nil
	interrupted bool            // Ctrl+C
	writeErr    error           // writing stdout failed
}

// sendStream sends text with the committed context and prints the answer
// while it streams (docs/tui-spec.md, section 14). Only StreamCompleted
// commits the exchange; incomplete, failed and canceled answers leave the
// conversation unchanged and are never sent as context again. Ctrl+C
// cancels the request and returns to the prompt. quit is set when the
// chat must end with code (ctx canceled, stdout closed).
func (c *chat) sendStream(ctx context.Context, text string) (code int, quit bool) {
	messages := c.Session.Context(text) // a snapshot: no lock is held below
	turn := c.receive(ctx, messages)

	if turn.writeErr != nil {
		return c.writeFailed(turn.writeErr)
	}
	if ctx.Err() != nil || turn.interrupted {
		if err := c.endLine(turn); err != nil {
			return c.writeFailed(err)
		}
		if ctx.Err() != nil {
			fmt.Fprintln(c.Stdout, "\nGoodbye.")
			return ExitInterrupted, true
		}
		fmt.Fprintln(c.Stderr, "\nRequest cancelled."+notSaved(turn))
		return 0, false
	}

	if completed, ok := turn.terminal.(llm.StreamCompleted); ok {
		var err error
		if !turn.printed {
			_, err = io.WriteString(c.Stdout, AIPrefix) // an empty answer still gets its prefix
		}
		if err == nil {
			_, err = io.WriteString(c.Stdout, "\n")
		}
		saveErr := c.Session.Commit(text, completed.Response) // the answer is complete either way
		if err != nil {
			return c.writeFailed(err)
		}
		c.completed(completed.Response, saveErr)
		return 0, false
	}

	if err := c.endLine(turn); err != nil {
		return c.writeFailed(err)
	}
	switch t := turn.terminal.(type) {
	case llm.StreamIncomplete:
		c.warn(fmt.Sprintf("Response incomplete: %s.%s", t.Reason, notSaved(turn)))
		err := c.Session.RecordIncomplete(llm.Response{
			Text: t.PartialText, IncompleteReason: t.Reason, Usage: t.Usage, ServiceTier: t.ServiceTier,
		})
		if err != nil {
			c.warn(err.Error())
		}
	case llm.StreamFailed:
		c.printError(t.Err.Error() + notSaved(turn))
	default: // a provider broke the contract
		c.printError("The response stream ended without a result." + notSaved(turn))
	}
	return 0, false
}

// receive runs one request: a worker iterates the provider stream into a
// bounded channel, and this loop, which owns stdout, prints the deltas.
// It returns after the first terminal event, Ctrl+C, a write error or the
// end of ctx, and only once the worker has finished (the stream and its
// connection are closed), so nothing of this request can reach the next.
func (c *chat) receive(ctx context.Context, messages []llm.Message) (turn streamTurn) {
	reqCtx, cancel := context.WithCancel(ctx)
	events := llm.StreamChannel(reqCtx, c.Stream(reqCtx, messages))
	defer func() {
		cancel()
		for range events { // wait for the worker; later events are dropped
		}
		if turn.interrupted {
			c.drainInterrupts() // Ctrl+C pressed again while canceling
		}
	}()
	for {
		select {
		case event, ok := <-events:
			if !ok {
				return turn // ended without a terminal event
			}
			switch e := event.(type) {
			case llm.TextDelta:
				if turn.writeErr = c.writeDelta(&turn, e.Text); turn.writeErr != nil {
					return turn
				}
			case llm.ReasoningDelta:
				// thinking is never shown
			default:
				turn.terminal = event
				return turn
			}
		case <-c.Interrupts:
			turn.interrupted = true
			return turn
		case <-ctx.Done():
			return turn
		}
	}
}

// writeDelta prints a piece of the answer, with "AI > " before the first.
func (c *chat) writeDelta(turn *streamTurn, text string) error {
	if !turn.printed {
		if _, err := io.WriteString(c.Stdout, AIPrefix); err != nil {
			return err
		}
		turn.printed = true
	}
	_, err := io.WriteString(c.Stdout, text)
	return err
}

// completed reports a committed answer: usage with --usage, then a
// failed save.
func (c *chat) completed(resp llm.Response, saveErr error) {
	if c.ShowUsage {
		fmt.Fprintln(c.Stderr, FormatUsage(resp.Usage))
		if line := FormatDuration(resp.Timing); line != "" {
			fmt.Fprintln(c.Stderr, line)
		}
	}
	if saveErr != nil {
		c.warn(saveErr.Error())
	}
}

// endLine ends the line of a partial answer.
func (c *chat) endLine(turn streamTurn) error {
	if !turn.printed {
		return nil
	}
	_, err := io.WriteString(c.Stdout, "\n")
	return err
}

// writeFailed ends the chat after stdout failed: quietly with 141 when
// the reader went away (nothing more is written to stdout), else with an
// error on stderr.
func (c *chat) writeFailed(err error) (int, bool) {
	if errors.Is(err, syscall.EPIPE) {
		return ExitBrokenPipe, true
	}
	c.printError(fmt.Sprintf("Could not write the answer: %v", err))
	return ExitError, true
}

func notSaved(turn streamTurn) string {
	if turn.printed {
		return " The partial response was not saved."
	}
	return ""
}
