package chat

import (
	"context"
	"fmt"
	"slices"
	"strings"

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
// the model supports it, Ollama thinking off). The model's support is
// looked up in any case; a failed lookup is a notice.
func ResolveSettings(ctx context.Context, p llm.Provider, model string, saved storage.Settings, o Overrides) (Settings, []string, error) {
	return resolveSettings(ctx, p, model, saved, o, true)
}

// ResolveSwitchedSettings is ResolveSettings for a model chosen in the
// fullscreen UI: an override the model does not support is skipped with a
// notice instead of being an error.
func ResolveSwitchedSettings(ctx context.Context, p llm.Provider, model string, saved storage.Settings, o Overrides) (Settings, []string, error) {
	return resolveSettings(ctx, p, model, saved, o, false)
}

func resolveSettings(ctx context.Context, p llm.Provider, model string, saved storage.Settings, o Overrides, strict bool) (Settings, []string, error) {
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

	// A lenient override the model does not support falls back to the
	// saved value, as if it had not been given.
	decided := false
	if o.Reasoning != "" {
		s, err := getSupport()
		if err != nil {
			return Settings{}, nil, err
		}
		settings.Reasoning, err = llm.ResolveReasoning(model, s, o.Reasoning)
		switch {
		case err == nil:
			decided = true
		case strict:
			return Settings{}, nil, &SettingsError{err}
		default:
			notices = append(notices, strings.TrimSuffix(err.Error(), ".")+"; using the default.")
		}
	}
	if !decided && prefs.Reasoning != "" {
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
	} else if !decided {
		// The built-in default depends on the model's support; as in the
		// Python edition, a failed lookup is a notice. (The lookup is
		// cached, so requests do not ask again.)
		if _, err := getSupport(); err != nil {
			return Settings{}, nil, err
		}
	}

	decided = false
	if o.MaxOutputTokens > 0 {
		limit, err := outputLimit(ctx, p, model)
		if err != nil {
			return Settings{}, nil, err
		}
		err = llm.CheckMaxOutputTokens(model, o.MaxOutputTokens, limit)
		switch {
		case err == nil:
			settings.MaxOutputTokens, decided = o.MaxOutputTokens, true
		case strict:
			return Settings{}, nil, &SettingsError{err}
		default:
			notices = append(notices, err.Error()+" Using the default.")
		}
	}
	if !decided && prefs.MaxOutputTokensSet && prefs.MaxOutputTokens > 0 { // 0: saved "model default"
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

// EffectiveReasoning is the reasoning value requests with these settings
// send, in the provider's vocabulary, given the model's support: the
// chosen value, else NeuroBräu's built-in default. "" sends no parameter,
// so the model uses its own default.
func (s Settings) EffectiveReasoning(support llm.ReasoningSupport) string {
	if s.Reasoning.Explicit {
		return s.Reasoning.Effort
	}
	return support.Default
}
