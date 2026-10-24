package chat

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/pricing"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// FormatUsage is the --usage line: the reported input, output and total
// counts only.
func FormatUsage(u *llm.Usage) string {
	if u == nil {
		return "Token usage: not reported"
	}
	var parts []string
	for _, f := range []struct {
		name  string
		count *int
	}{{"input", u.InputTokens}, {"output", u.OutputTokens}, {"total", u.TotalTokens}} {
		if f.count != nil {
			parts = append(parts, fmt.Sprintf("%s=%d", f.name, *f.count))
		}
	}
	if len(parts) == 0 {
		return "Token usage: not reported"
	}
	return "Token usage: " + strings.Join(parts, ", ")
}

// TotalUsage sums each count over the records; a count no record reported
// stays nil (unknown, not zero). Nil if there are no records.
func TotalUsage(records []storage.UsageRecord) *llm.Usage {
	var total *llm.Usage
	for _, r := range records {
		if total == nil {
			total = &llm.Usage{}
		}
		for _, f := range []struct {
			sum **int
			add *int
		}{
			{&total.InputTokens, r.Usage.InputTokens},
			{&total.OutputTokens, r.Usage.OutputTokens},
			{&total.TotalTokens, r.Usage.TotalTokens},
			{&total.CachedInputTokens, r.Usage.CachedInputTokens},
			{&total.CacheWriteTokens, r.Usage.CacheWriteTokens},
			{&total.ReasoningTokens, r.Usage.ReasoningTokens},
		} {
			if f.add == nil {
				continue
			}
			n := *f.add
			if *f.sum != nil {
				n += **f.sum
			}
			*f.sum = &n
		}
	}
	return total
}

// UsageReport is the /usage text of the plain and the fullscreen chat: the
// conversation's API usage as reported and its ESTIMATED cost
// (docs/tui-spec.md, section 15). Cached and cache-write tokens are shown
// as parts of the input, reasoning tokens as part of the output.
func UsageReport(records []storage.UsageRecord) string {
	if len(records) == 0 {
		return "Token usage: no API requests in this conversation yet."
	}
	lines := []string{fmt.Sprintf("Conversation usage (%d request(s), as reported by the API):", len(records))}
	lines = append(lines, usageBreakdown(TotalUsage(records))...)
	incomplete := 0
	var latest *storage.UsageRecord
	for i, r := range records {
		if r.Status == storage.StatusIncomplete {
			incomplete++
		} else {
			latest = &records[i]
		}
	}
	if latest != nil && latest.Usage.InputTokens != nil {
		lines = append(lines, fmt.Sprintf("Latest request input: %s tokens "+
			"(the whole history is sent, and billed, with every request)", formatCount(*latest.Usage.InputTokens)))
	}
	if incomplete > 0 {
		lines = append(lines, fmt.Sprintf("Includes %d incomplete answer(s): billed, but not saved as messages.", incomplete))
	}
	lines = append(lines, pricing.Describe(pricing.Total(records)))
	return strings.Join(lines, "\n")
}

// usageBreakdown has one `label: count` line per reported count: none for
// records that report no counts, as in the Python edition.
func usageBreakdown(u *llm.Usage) []string {
	if u == nil {
		return []string{"Token usage: not reported"}
	}
	var lines []string
	for _, f := range []struct {
		label string
		count *int
	}{
		{"Input tokens", u.InputTokens},
		{"  of which cached", u.CachedInputTokens},
		{"  of which cache writes", u.CacheWriteTokens},
		{"Output tokens", u.OutputTokens},
		{"  of which reasoning", u.ReasoningTokens},
		{"Total tokens", u.TotalTokens},
	} {
		if f.count != nil {
			lines = append(lines, f.label+": "+formatCount(*f.count))
		}
	}
	return lines
}

// formatCount writes n with thousands separators, e.g. 1,234,567.
func formatCount(n int) string {
	s := strconv.Itoa(n)
	if n < 0 {
		return "-" + formatCount(-n)
	}
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

// FormatDuration is the `Duration: …` line for provider-reported timing
// (Ollama); "" when the total is not reported.
func FormatDuration(t llm.Timing) string {
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
