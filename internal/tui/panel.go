package tui

import (
	"fmt"
	"strconv"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/pricing"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// providerLabels are the display names of the providers.
var providerLabels = map[string]string{"openai": "OpenAI", "ollama": "Ollama"}

func providerLabel(name string) string {
	if label, ok := providerLabels[name]; ok {
		return label
	}
	return name
}

// panelRow is one row of the settings panel.
type panelRow struct {
	text string
	role role
}

// settingsPanel returns the rows of the LLM SETTINGS panel
// (docs/tui-spec.md, section 4a). Token counts are the totals of the
// conversation's usage records. reasoning is the effective value in the
// provider's vocabulary ("" sends no parameter). The cost is the
// conversation's estimate in compact form.
func settingsPanel(sel storage.Selection, reasoning string, s chat.Settings, records []storage.UsageRecord) []panelRow {
	value := func(text string) panelRow { return panelRow{fit(" "+text, panelContent), roleText} }
	// Exact numbers when they fit, abbreviated otherwise.
	counts := func(exact, short string) panelRow {
		if len([]rune(exact))+1 <= panelContent {
			return value(exact)
		}
		return value(short)
	}
	heading := func(text string) panelRow { return panelRow{text, roleBorder} }

	if reasoning == "" {
		reasoning = "model default"
	}
	maxTokens := "default"
	if s.MaxOutputTokens > 0 {
		maxTokens = strconv.Itoa(s.MaxOutputTokens)
	}
	style := s.AnswerStyle
	if style == "" {
		style = "default"
	}
	rows := []panelRow{
		heading("F2 MODEL"), value(sel.Model),
		heading("F3 PROVIDER"), value(providerLabel(sel.Provider)),
		heading("F4 REASONING"), value(reasoning),
		heading("F5 MAX TOKENS"), value(maxTokens),
		heading("F6 ANSWER STYLE"), value(style),
		heading("TOKENS"),
	}
	if u := chat.TotalUsage(records); u == nil {
		rows = append(rows, value("-"))
	} else {
		rows = append(rows,
			counts(fmt.Sprintf("in %s out %s", exact(u.InputTokens), exact(u.OutputTokens)),
				fmt.Sprintf("in %s out %s", compact(u.InputTokens), compact(u.OutputTokens))),
			counts("total "+exact(u.TotalTokens), "total "+compact(u.TotalTokens)))
		if u.CachedInputTokens != nil && *u.CachedInputTokens > 0 {
			rows = append(rows, value("cached "+compact(u.CachedInputTokens)))
		}
		if u.ReasoningTokens != nil && *u.ReasoningTokens > 0 {
			rows = append(rows, value("reasoning "+compact(u.ReasoningTokens)))
		}
	}
	if context := latestContext(records); context != nil {
		rows = append(rows, value("context "+compact(context)))
	}
	return append(rows, heading("COST (EST.)"), value(pricing.Short(pricing.Total(records))))
}

// latestContext is the input tokens of the latest completed request: what
// the conversation currently costs per request.
func latestContext(records []storage.UsageRecord) *int {
	for i := len(records) - 1; i >= 0; i-- {
		if records[i].Status == storage.StatusCompleted {
			return records[i].Usage.InputTokens
		}
	}
	return nil
}

func exact(n *int) string {
	if n == nil {
		return "?"
	}
	return strconv.Itoa(*n)
}

// compact abbreviates a count with three significant digits (12.3k,
// 4.56M, 2G); numbers below 10 000 are never abbreviated.
func compact(n *int) string {
	if n == nil {
		return "?"
	}
	units := []struct {
		size   float64
		suffix string
	}{{1e9, "G"}, {1e6, "M"}, {1e3, "k"}}
	for i, u := range units {
		if *n < 10_000 || float64(*n) < u.size {
			continue
		}
		v := float64(*n) / u.size
		s := strconv.FormatFloat(v, 'g', 3, 64)
		if f, _ := strconv.ParseFloat(s, 64); f >= 1000 && i > 0 {
			// 999 999 rounds to 1000k: use the next unit instead.
			s = strconv.FormatFloat(float64(*n)/units[i-1].size, 'g', 3, 64)
			return s + units[i-1].suffix
		}
		return s + u.suffix
	}
	return strconv.Itoa(*n)
}
