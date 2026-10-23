package llm

import (
	"fmt"
	"slices"
	"strings"
)

// Reasoning is the reasoning setting for one request, in the provider's
// vocabulary (OpenAI efforts such as "low"; Ollama "off"/"on" or named
// levels).
//
//   - Reasoning{}                         no choice: the provider's built-in
//     default (OpenAI: "none" where the model supports it; Ollama: thinking
//     off on thinking models), which may be dropped if the API rejects it
//   - Reasoning{Explicit: true}           the user chose "model default":
//     send no reasoning parameter
//   - Reasoning{Effort: v, Explicit: true} the user chose v: send it, and
//     report a rejection instead of dropping it
type Reasoning struct {
	Effort   string
	Explicit bool
}

// ReasoningSupport describes which reasoning values a model accepts, as far
// as the provider can confirm.
type ReasoningSupport struct {
	Values []string // selectable values, in the provider's vocabulary
	Known  bool     // false: support could not be confirmed
	// Default is NeuroBräu's built-in value when the user chose nothing;
	// "" sends no reasoning parameter.
	Default string
	// NoModelDefault is set when "send no parameter" is not a choice (Ollama
	// models that only switch thinking off or on).
	NoModelDefault bool
	// ModelDefault is what the model itself uses without a parameter, if
	// documented (e.g. "medium" for gpt-6-luna).
	ModelDefault string
}

// Answer styles.
const (
	StyleBrief    = "brief"
	StyleNormal   = "normal"
	StyleDetailed = "detailed"
)

// AnswerStyles lists the answer styles in display order.
var AnswerStyles = []string{StyleBrief, StyleNormal, StyleDetailed}

// StyleInstructions returns the system/developer instruction for an answer
// style; "normal" (and any unknown style) adds none.
func StyleInstructions(style string) string {
	switch style {
	case StyleBrief:
		return "Answer briefly and concisely. Use a few sentences or a short list. " +
			"Leave out background and caveats unless the user asks for them."
	case StyleDetailed:
		return "Give a comprehensive, well-structured explanation. Cover the relevant " +
			"background and the reasoning behind the answer, and include concrete " +
			"examples when they help."
	default:
		return ""
	}
}

// ReasoningEfforts are the effort names accepted on the command line.
var ReasoningEfforts = []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}

// ModelDefault is the command-line value that sends no reasoning parameter,
// so the model uses its own default.
const ModelDefault = "model-default"

// ResolveReasoning turns a command-line reasoning value ("" when not given)
// into a request setting for a model. A value the model is known not to
// support is an error.
func ResolveReasoning(model string, support ReasoningSupport, requested string) (Reasoning, error) {
	switch {
	case requested == "":
		return Reasoning{}, nil
	case requested == ModelDefault:
		if support.NoModelDefault {
			return Reasoning{}, fmt.Errorf("%s has no model-default reasoning choice (%s).",
				model, DescribeSupport(model, support))
		}
		return Reasoning{Explicit: true}, nil
	case !support.Known:
		return Reasoning{Effort: requested, Explicit: true}, nil // support unknown: let the API decide
	case len(support.Values) == 0 && requested == "none":
		return Reasoning{Explicit: true}, nil // "no reasoning" on a model without it: send nothing
	}
	if mapped := mapReasoning(support.Values, requested); mapped != "" {
		return Reasoning{Effort: mapped, Explicit: true}, nil
	}
	return Reasoning{}, fmt.Errorf("%s; '%s' is not available.", DescribeSupport(model, support), requested)
}

// mapReasoning translates an effort name into a value the model accepts:
// "none" and "off" are the same choice, and any other effort selects "on"
// on models that only switch thinking off or on.
func mapReasoning(values []string, requested string) string {
	if slices.Contains(values, requested) {
		return requested
	}
	alias := map[string]string{"none": "off", "off": "none"}[requested]
	if alias != "" && slices.Contains(values, alias) {
		return alias
	}
	if requested != "none" && requested != "off" && slices.Contains(values, "on") {
		return "on"
	}
	return ""
}

// DescribeSupport says which reasoning values a model supports.
func DescribeSupport(model string, support ReasoningSupport) string {
	if len(support.Values) == 0 {
		return model + " does not support reasoning"
	}
	labels := make([]string, len(support.Values))
	for i, v := range support.Values {
		labels[i] = reasoningLabel(v)
	}
	return model + " supports reasoning: " + strings.Join(labels, ", ")
}

func reasoningLabel(value string) string {
	switch value {
	case "xhigh":
		return "XHigh"
	case "on":
		return "On"
	case "off":
		return "Off"
	case "":
		return "Model default"
	}
	return strings.ToUpper(value[:1]) + value[1:]
}

// CheckMaxOutputTokens reports a value above a known limit (0: unknown).
func CheckMaxOutputTokens(model string, value, limit int) error {
	if value <= 0 {
		return fmt.Errorf("Max output tokens must be a positive whole number, not %d.", value)
	}
	if limit > 0 && value > limit {
		return fmt.Errorf("%d exceeds the limit of %d output tokens for %s.", value, limit, model)
	}
	return nil
}
