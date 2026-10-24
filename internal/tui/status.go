package tui

import (
	"fmt"
	"strings"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// Runtime states (docs/tui-spec.md, section 5).
const (
	stateReady      = "READY"
	stateConnecting = "CONNECTING"
	stateGenerating = "GENERATING"
	stateDone       = "DONE"
	stateCancelled  = "CANCELLED"
	stateError      = "ERROR"
	stateWarning    = "WARNING"
)

const (
	statusSeparator = " | "
	cancelHint      = "Ctrl+C Cancel"
)

// status is what the runtime bar shows: the state word, then details.
type status struct {
	state string
	parts []string
}

func (s status) String() string {
	return strings.Join(append([]string{s.state}, s.parts...), statusSeparator)
}

// transient states last until the next key press, then the bar returns
// to READY.
func (s status) transient() bool {
	switch s.state {
	case stateDone, stateCancelled, stateError, stateWarning:
		return true
	}
	return false
}

func (s status) role() role {
	switch s.state {
	case stateConnecting, stateGenerating, stateCancelled, stateWarning:
		return roleWaiting
	case stateDone:
		return roleAI
	case stateError:
		return roleError
	}
	return roleText
}

func ready(parts ...string) status { return status{stateReady, parts} }

func warning(text string) status { return status{stateWarning, []string{text}} }

func seconds(d time.Duration) string { return fmt.Sprintf("%.2fs", d.Seconds()) }

// activeStatus is CONNECTING or GENERATING with the client-observed
// elapsed time.
func activeStatus(state string, elapsed time.Duration, detail string) status {
	var parts []string
	if detail != "" {
		parts = append(parts, detail)
	}
	return status{state, append(parts, fmt.Sprintf("%.1fs elapsed", elapsed.Seconds()), cancelHint)}
}

// doneStatus shows the durations the provider reported (Ollama), else the
// latency this client observed, labeled as such. Missing values are left
// out, never estimated; no speed is derived from client time. firstText
// is 0 when no text arrived or the answer was not streamed.
func doneStatus(resp llm.Response, observed, firstText time.Duration) status {
	t := resp.Timing
	if t.Total > 0 {
		parts := []string{seconds(t.Total) + " total"}
		if t.Load > 0 {
			parts = append(parts, "load "+seconds(t.Load))
		}
		if t.Eval > 0 {
			parts = append(parts, "generation "+seconds(t.Eval))
			if resp.Usage != nil && resp.Usage.OutputTokens != nil {
				parts = append(parts, fmt.Sprintf("%.2f tok/s", float64(*resp.Usage.OutputTokens)/t.Eval.Seconds()))
			}
		}
		return status{stateDone, parts}
	}
	parts := []string{seconds(observed) + " observed"}
	if firstText > 0 {
		parts = append(parts, "first text after "+seconds(firstText))
	}
	return status{stateDone, parts}
}
