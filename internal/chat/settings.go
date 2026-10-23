package chat

import (
	"context"
	"fmt"
	"slices"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// Overrides are the generation settings given on the command line for
// this invocation; they are never saved. Zero values: not given.
type Overrides struct {
	Reasoning       string // an effort name or llm.ModelDefault
	MaxOutputTokens int
	AnswerStyle     string
}

// Settings are the effective generation settings of the chat.
type Settings struct {
	Reasoning       llm.Reasoning
	MaxOutputTokens int // 0: the model's default
	AnswerStyle     string
}

// SettingsError is a command-line setting the model is known not to
// support (exit code 2 before the chat starts).
type SettingsError struct{ Err error }

func (e *SettingsError) Error() string { return e.Err.Error() }
func (e *SettingsError) Unwrap() error { return e.Err }

// ResolveSettings works out the settings for one provider/model: CLI
// overrides, else the saved preferences, else NeuroBräu's built-in
// defaults (docs/tui-spec.md, section 13). An unsupported override is a
// *SettingsError; an unsupported saved value is skipped with a notice.
// Only a canceled ctx is returned as another error.
//
// Without an override or a usable saved value, reasoning stays
// llm.Reasoning{}: the provider applies its built-in default ("none" where
// the model supports it, Ollama thinking off).
func ResolveSettings(ctx context.Context, p llm.Provider, model string, saved storage.Settings, o Overrides) (Settings, []string, error) {
	var notices []string
	var support *llm.ReasoningSupport
	getSupport := func() (llm.ReasoningSupport, error) {
		if support == nil {
			s, err := p.ReasoningSupport(ctx, model)
			if err != nil {
				if ctx.Err() != nil {
					return s, err
				}
				notices = append(notices, fmt.Sprintf("Could not check reasoning support for %s: %v", model, err))
				s = llm.ReasoningSupport{} // unknown: explicit values are passed through
			}
			support = &s
		}
		return *support, nil
	}

	prefs := saved.ForModel(p.Name(), model)
	settings := Settings{AnswerStyle: saved.AnswerStyle}
	if o.AnswerStyle != "" {
		settings.AnswerStyle = o.AnswerStyle
	}
	if settings.AnswerStyle == "" {
		settings.AnswerStyle = llm.StyleNormal
	}

	switch {
	case o.Reasoning != "":
		s, err := getSupport()
		if err != nil {
			return Settings{}, nil, err
		}
		if settings.Reasoning, err = llm.ResolveReasoning(model, s, o.Reasoning); err != nil {
			return Settings{}, nil, &SettingsError{err}
		}
	case prefs.Reasoning != "":
		s, err := getSupport()
		if err != nil {
			return Settings{}, nil, err
		}
		switch {
		case prefs.Reasoning == storage.ReasoningDefault && !s.NoModelDefault:
			settings.Reasoning = llm.Reasoning{Explicit: true} // send no parameter
		case s.Known && slices.Contains(s.Values, prefs.Reasoning):
			settings.Reasoning = llm.Reasoning{Effort: prefs.Reasoning, Explicit: true}
		default:
			notices = append(notices, fmt.Sprintf("Saved reasoning '%s' is not available for %s (%s); using %s.",
				llm.ReasoningLabel(prefs.Reasoning), model, llm.DescribeSupport(model, s), llm.ReasoningLabel(s.Default)))
		}
	}

	switch {
	case o.MaxOutputTokens > 0:
		limit, err := outputLimit(ctx, p, model)
		if err != nil {
			return Settings{}, nil, err
		}
		if err := llm.CheckMaxOutputTokens(model, o.MaxOutputTokens, limit); err != nil {
			return Settings{}, nil, &SettingsError{err}
		}
		settings.MaxOutputTokens = o.MaxOutputTokens
	case prefs.MaxOutputTokensSet && prefs.MaxOutputTokens > 0: // 0: saved "model default"
		limit, err := outputLimit(ctx, p, model)
		if err != nil {
			return Settings{}, nil, err
		}
		if err := llm.CheckMaxOutputTokens(model, prefs.MaxOutputTokens, limit); err != nil {
			notices = append(notices, fmt.Sprintf("Saved setting ignored: %v Using the default.", err))
		} else {
			settings.MaxOutputTokens = prefs.MaxOutputTokens
		}
	}
	return settings, notices, nil
}

// outputLimit returns the model's known output limit, or 0. A lookup
// error (e.g. a missing model, which the chat request reports) does not
// restrict the value; only cancellation is returned.
func outputLimit(ctx context.Context, p llm.Provider, model string) (int, error) {
	limit, err := p.MaxOutputLimit(ctx, model)
	if err != nil {
		if ctx.Err() != nil {
			return 0, err
		}
		return 0, nil
	}
	return limit, nil
}
