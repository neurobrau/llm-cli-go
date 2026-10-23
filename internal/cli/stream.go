package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"syscall"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// flusher is a writer with buffered output (e.g. bufio.Writer); os.Stdout
// writes immediately.
type flusher interface{ Flush() error }

// streamPrompt prints a one-shot answer while it streams
// (docs/tui-spec.md, section 14). Text deltas go to stdout as they arrive;
// reasoning is never printed. A final newline follows any text and an
// empty completed answer, before anything goes to stderr. Exit codes:
// completed 0, incomplete 8, failed by error kind, Ctrl+C 130, stdout
// closed by the reader 141 (the stream is closed at once).
func streamPrompt(ctx context.Context, provider llm.StreamingProvider, req llm.Request, showUsage bool,
	stdout, stderr io.Writer) int {
	printed := false
	var terminal llm.StreamEvent
	var writeErr error
	for event := range provider.Stream(ctx, req) {
		if delta, ok := event.(llm.TextDelta); ok {
			if writeErr = write(stdout, delta.Text); writeErr != nil {
				break // closes the stream and its connection
			}
			printed = true
			continue
		}
		if _, ok := event.(llm.ReasoningDelta); !ok {
			terminal = event
		}
	}
	if writeErr != nil {
		return writeFailed(stderr, writeErr)
	}

	// After Ctrl+C the outcome is "interrupted", whatever the stream said.
	interrupted := ctx.Err() != nil
	_, completed := terminal.(llm.StreamCompleted)
	if printed || completed && !interrupted {
		if err := write(stdout, "\n"); err != nil {
			return writeFailed(stderr, err)
		}
	}
	if interrupted {
		if printed {
			fmt.Fprintf(stderr, "%s: interrupted\n", ProgName)
			return ExitInterrupted
		}
		return reportError(ctx, stderr, context.Canceled)
	}

	switch t := terminal.(type) {
	case llm.StreamCompleted:
		if showUsage {
			fmt.Fprintln(stderr, chat.FormatUsage(t.Response.Usage))
			if line := formatDuration(t.Response.Timing); line != "" {
				fmt.Fprintln(stderr, line)
			}
		}
		return ExitOK
	case llm.StreamIncomplete:
		fmt.Fprintf(stderr, "%s: warning: response incomplete: %s.\n", ProgName, t.Reason)
		if showUsage && t.Usage != nil {
			fmt.Fprintln(stderr, chat.FormatUsage(t.Usage))
		}
		return ExitIncomplete
	case llm.StreamFailed:
		return reportError(ctx, stderr, t.Err)
	default: // a provider broke the contract
		return reportError(ctx, stderr, &llm.Error{Kind: llm.KindAPI, Message: "The response stream ended without a result."})
	}
}

// write writes s and flushes buffered output.
func write(w io.Writer, s string) error {
	if _, err := io.WriteString(w, s); err != nil {
		return err
	}
	if f, ok := w.(flusher); ok {
		return f.Flush()
	}
	return nil
}

// writeFailed handles a failed stdout write: quietly with 141 when the
// reader went away (e.g. `| head`), else as an error.
func writeFailed(stderr io.Writer, err error) int {
	if errors.Is(err, syscall.EPIPE) {
		return ExitBrokenPipe
	}
	fmt.Fprintf(stderr, "%s: error: could not write the answer: %v\n", ProgName, err)
	return llm.ExitCode(&llm.Error{Kind: llm.KindAPI})
}

// formatDuration is the `Duration: …` line for provider-reported timing
// (Ollama); "" when the total is not reported.
func formatDuration(t llm.Timing) string {
	if t.Total <= 0 {
		return ""
	}
	line := fmt.Sprintf("Duration: %.2f s", t.Total.Seconds())
	var details []string
	if t.Load > 0 {
		details = append(details, fmt.Sprintf("load %.2f s", t.Load.Seconds()))
	}
	if t.Eval > 0 {
		details = append(details, fmt.Sprintf("generation %.2f s", t.Eval.Seconds()))
	}
	if len(details) > 0 {
		line += " (" + strings.Join(details, ", ") + ")"
	}
	return line
}
