package storage

import (
	"encoding/json"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// Usage record statuses.
const (
	StatusCompleted  = "completed"
	StatusIncomplete = "incomplete" // billed, but the answer is not a message
)

// UsageRecord is the reported usage of one API request (docs/tui-spec.md,
// section 15). Counts are stored exactly as reported: a nil count was not
// reported. CachedInputTokens and CacheWriteTokens are parts of
// InputTokens; ReasoningTokens is part of OutputTokens.
type UsageRecord struct {
	At          time.Time
	Provider    string
	Model       string
	Status      string // StatusCompleted or StatusIncomplete
	ServiceTier string // "" if not reported
	Usage       llm.Usage
}

// usageRecordJSON fixes the key order of a written record.
type usageRecordJSON struct {
	At                string `json:"at"`
	Provider          string `json:"provider"`
	Model             string `json:"model"`
	Status            string `json:"status"`
	ServiceTier       string `json:"service_tier,omitempty"`
	InputTokens       *int   `json:"input_tokens,omitempty"`
	OutputTokens      *int   `json:"output_tokens,omitempty"`
	TotalTokens       *int   `json:"total_tokens,omitempty"`
	CachedInputTokens *int   `json:"cached_input_tokens,omitempty"`
	CacheWriteTokens  *int   `json:"cache_write_tokens,omitempty"`
	ReasoningTokens   *int   `json:"reasoning_tokens,omitempty"`
}

func (r UsageRecord) toJSON() usageRecordJSON {
	status := r.Status
	if status == "" {
		status = StatusCompleted
	}
	u := r.Usage
	return usageRecordJSON{
		At: formatTimestamp(r.At), Provider: r.Provider, Model: r.Model, Status: status,
		ServiceTier: r.ServiceTier,
		InputTokens: u.InputTokens, OutputTokens: u.OutputTokens, TotalTokens: u.TotalTokens,
		CachedInputTokens: u.CachedInputTokens, CacheWriteTokens: u.CacheWriteTokens,
		ReasoningTokens: u.ReasoningTokens,
	}
}

// parseUsageRecord validates one stored record; ok is false if it is
// malformed, and the record is then skipped.
func parseUsageRecord(raw json.RawMessage) (UsageRecord, bool) {
	o, ok := asObject(raw)
	if !ok {
		return UsageRecord{}, false
	}
	at, err := parseTimestamp(o, "at")
	if err != nil {
		return UsageRecord{}, false
	}
	provider, okProvider := o.str("provider")
	model, okModel := o.str("model")
	status := StatusCompleted
	if _, found := o["status"]; found {
		status, _ = o.str("status") // null or a non-string is invalid below
	}
	if !okProvider || !okModel || (status != StatusCompleted && status != StatusIncomplete) {
		return UsageRecord{}, false
	}
	r := UsageRecord{At: at, Provider: provider, Model: model, Status: status}
	if o.present("service_tier") {
		if r.ServiceTier, ok = o.str("service_tier"); !ok {
			return UsageRecord{}, false
		}
	}
	for _, f := range []struct {
		key string
		dst **int
	}{
		{"input_tokens", &r.Usage.InputTokens},
		{"output_tokens", &r.Usage.OutputTokens},
		{"total_tokens", &r.Usage.TotalTokens},
		{"cached_input_tokens", &r.Usage.CachedInputTokens},
		{"cache_write_tokens", &r.Usage.CacheWriteTokens},
		{"reasoning_tokens", &r.Usage.ReasoningTokens},
	} {
		n, found, err := o.count(f.key)
		if err != nil {
			return UsageRecord{}, false
		}
		if found {
			*f.dst = &n
		}
	}
	return r, true
}
