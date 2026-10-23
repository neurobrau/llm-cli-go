package openai

import (
	"regexp"
	"slices"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// knownModel holds confirmed capabilities. The Models API does not report
// them, so only models listed here get confirmed reasoning choices.
type knownModel struct {
	efforts         []string // empty: known not to accept the reasoning parameter
	maxOutputTokens int      // 0: unknown
}

// knownModels is the table from the Python reference (KNOWN_MODELS). Dated
// snapshots ("gpt-5-2025-08-07") share their base model's entry.
var knownModels = map[string]knownModel{
	// The project's default model. Without a choice NeuroBräu sends "none";
	// the model's own documented default is "medium".
	"gpt-6-luna":   {[]string{"none", "low", "medium", "high", "xhigh", "max"}, 0},
	"gpt-5.2":      {[]string{"none", "low", "medium", "high", "xhigh"}, 128_000},
	"gpt-5.1":      {[]string{"none", "low", "medium", "high"}, 128_000},
	"gpt-5":        {[]string{"minimal", "low", "medium", "high"}, 128_000},
	"gpt-5-mini":   {[]string{"minimal", "low", "medium", "high"}, 128_000},
	"gpt-5-nano":   {[]string{"minimal", "low", "medium", "high"}, 128_000},
	"o3":           {[]string{"low", "medium", "high"}, 100_000},
	"o3-mini":      {[]string{"low", "medium", "high"}, 100_000},
	"o4-mini":      {[]string{"low", "medium", "high"}, 100_000},
	"gpt-4.1":      {nil, 32_768},
	"gpt-4.1-mini": {nil, 32_768},
	"gpt-4.1-nano": {nil, 32_768},
	"gpt-4o":       {nil, 16_384},
	"gpt-4o-mini":  {nil, 16_384},
}

// modelDefaultEfforts is what a model uses when no reasoning parameter is
// sent, where documented. It is information only, not NeuroBräu's default.
var modelDefaultEfforts = map[string]string{"gpt-6-luna": "medium"}

var snapshotSuffix = regexp.MustCompile(`-\d{4}-\d{2}-\d{2}$`)

// baseModel strips a dated snapshot suffix.
func baseModel(model string) string { return snapshotSuffix.ReplaceAllString(model, "") }

func lookupModel(model string) (knownModel, bool) {
	if m, ok := knownModels[model]; ok {
		return m, true
	}
	m, ok := knownModels[baseModel(model)]
	return m, ok
}

// reasoningSupportFor returns the confirmed efforts from the table; unknown
// models get none confirmed and no built-in default.
func reasoningSupportFor(model string) llm.ReasoningSupport {
	m, ok := lookupModel(model)
	if !ok {
		return llm.ReasoningSupport{}
	}
	support := llm.ReasoningSupport{Values: slices.Clone(m.efforts), Known: true}
	if slices.Contains(m.efforts, "none") {
		support.Default = "none"
	}
	support.ModelDefault = modelDefaultEfforts[model]
	if support.ModelDefault == "" {
		support.ModelDefault = modelDefaultEfforts[baseModel(model)]
	}
	return support
}
