package chat

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

var (
	lunaSupport = llm.ReasoningSupport{Values: []string{"none", "low", "medium", "high", "xhigh"}, Known: true,
		Default: "none", ModelDefault: "medium"}
	onOffSupport = llm.ReasoningSupport{Values: []string{"off", "on"}, Known: true, Default: "off", NoModelDefault: true}
)

func savedFor(style string, prefs storage.ModelPreferences) storage.Settings {
	s := storage.DefaultSettings()
	if style != "" {
		s.AnswerStyle = style
	}
	return s.WithModel("openai", "m", prefs)
}

func TestResolveSettingsPrecedence(t *testing.T) {
	saved := savedFor("detailed", storage.ModelPreferences{Reasoning: "high", MaxOutputTokensSet: true, MaxOutputTokens: 600})
	tests := []struct {
		name      string
		saved     storage.Settings
		overrides Overrides
		want      Settings
	}{
		{"built-in defaults", storage.DefaultSettings(), Overrides{}, Settings{AnswerStyle: "normal"}},
		{"saved", saved, Overrides{}, Settings{Reasoning: llm.Reasoning{Effort: "high", Explicit: true}, MaxOutputTokens: 600, AnswerStyle: "detailed"}},
		{"CLI over saved", saved, Overrides{Reasoning: "low", MaxOutputTokens: 100, AnswerStyle: "brief"},
			Settings{Reasoning: llm.Reasoning{Effort: "low", Explicit: true}, MaxOutputTokens: 100, AnswerStyle: "brief"}},
		{"CLI model-default", saved, Overrides{Reasoning: llm.ModelDefault},
			Settings{Reasoning: llm.Reasoning{Explicit: true}, MaxOutputTokens: 600, AnswerStyle: "detailed"}},
		{"saved default and null", savedFor("", storage.ModelPreferences{Reasoning: "default", MaxOutputTokensSet: true}), Overrides{},
			Settings{Reasoning: llm.Reasoning{Explicit: true}, AnswerStyle: "normal"}},
		{"other model's preferences", saved.WithModel("openai", "m", storage.ModelPreferences{}).WithModel("openai", "other",
			storage.ModelPreferences{Reasoning: "high"}), Overrides{}, Settings{AnswerStyle: "detailed"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &fakeProvider{support: lunaSupport, limit: 1000}
			got, notices, err := ResolveSettings(context.Background(), p, "m", tt.saved, tt.overrides)
			if err != nil || notices != nil || !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %+v, %q, %v; want %+v", got, notices, err, tt.want)
			}
		})
	}
}

func TestUnsupportedSavedValuesFallBack(t *testing.T) {
	tests := []struct {
		name    string
		support llm.ReasoningSupport
		prefs   storage.ModelPreferences
		notice  string
	}{
		{"reasoning level", onOffSupport, storage.ModelPreferences{Reasoning: "high"},
			"Saved reasoning 'High' is not available for m (m supports reasoning: Off, On); using Off."},
		{"default on an on/off model", onOffSupport, storage.ModelPreferences{Reasoning: "default"},
			"Saved reasoning 'Default' is not available for m (m supports reasoning: Off, On); using Off."},
		{"unknown support", llm.ReasoningSupport{}, storage.ModelPreferences{Reasoning: "low"},
			"Saved reasoning 'Low' is not available for m (m does not support reasoning); using Model default."},
		{"max tokens above the limit", lunaSupport, storage.ModelPreferences{MaxOutputTokensSet: true, MaxOutputTokens: 5000},
			"Saved setting ignored: 5000 exceeds the limit of 4096 output tokens for m. Using the default."},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &fakeProvider{support: tt.support, limit: 4096}
			got, notices, err := ResolveSettings(context.Background(), p, "m", savedFor("", tt.prefs), Overrides{})
			if err != nil || !reflect.DeepEqual(got, Settings{AnswerStyle: "normal"}) || !reflect.DeepEqual(notices, []string{tt.notice}) {
				t.Errorf("got %+v, %q, %v", got, notices, err)
			}
		})
	}
}

func TestUnsupportedCLIValuesAreErrors(t *testing.T) {
	tests := []struct {
		name      string
		support   llm.ReasoningSupport
		overrides Overrides
		want      string
	}{
		{"effort", lunaSupport, Overrides{Reasoning: "minimal"}, "m supports reasoning: None, Low, Medium, High, XHigh; 'minimal' is not available."},
		{"model-default on on/off", onOffSupport, Overrides{Reasoning: llm.ModelDefault},
			"m has no model-default reasoning choice (m supports reasoning: Off, On)."},
		{"max tokens", lunaSupport, Overrides{MaxOutputTokens: 5000}, "5000 exceeds the limit of 4096 output tokens for m."},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &fakeProvider{support: tt.support, limit: 4096}
			// A valid saved value does not rescue an invalid flag.
			saved := savedFor("", storage.ModelPreferences{Reasoning: "off", MaxOutputTokensSet: true, MaxOutputTokens: 10})
			_, _, err := ResolveSettings(context.Background(), p, "m", saved, tt.overrides)
			var settingsErr *SettingsError
			if !errors.As(err, &settingsErr) || err.Error() != tt.want {
				t.Errorf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestSupportLookupFailure(t *testing.T) {
	p := &fakeProvider{supErr: &llm.Error{Kind: llm.KindConnection, Message: "Could not connect to Ollama."}}
	got, notices, err := ResolveSettings(context.Background(), p, "m", storage.DefaultSettings(), Overrides{Reasoning: "high"})
	if err != nil || got.Reasoning != (llm.Reasoning{Effort: "high", Explicit: true}) ||
		len(notices) != 1 || !strings.HasPrefix(notices[0], "Could not check reasoning support for m: Could not connect") {
		t.Errorf("got %+v, %q, %v", got, notices, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p = &fakeProvider{supErr: context.Canceled}
	if _, _, err := ResolveSettings(ctx, p, "m", storage.DefaultSettings(), Overrides{Reasoning: "high"}); !errors.Is(err, context.Canceled) {
		t.Errorf("canceled lookup: %v", err)
	}
}

func TestUsageReport(t *testing.T) {
	at := time.Date(2026, 10, 22, 5, 54, 0, 0, time.UTC)
	records := []storage.UsageRecord{
		{At: at, Status: "completed", Usage: llm.Usage{InputTokens: intp(1000), OutputTokens: intp(200), TotalTokens: intp(1200),
			CachedInputTokens: intp(600), CacheWriteTokens: intp(100), ReasoningTokens: intp(50)}},
		{At: at, Status: "completed", Usage: llm.Usage{InputTokens: intp(1250), OutputTokens: intp(0)}},
		{At: at, Status: "incomplete", Usage: llm.Usage{InputTokens: intp(1300), OutputTokens: intp(4000), ReasoningTokens: intp(0)}},
	}
	want := `Conversation usage (3 request(s), as reported by the API):
Input tokens: 3,550
  of which cached: 600
  of which cache writes: 100
Output tokens: 4,200
  of which reasoning: 50
Total tokens: 1,200
Latest request input: 1,250 tokens (the whole history is sent, and billed, with every request)
Includes 1 incomplete answer(s): billed, but not saved as messages.
Estimated cost: not implemented in the Go edition yet.`
	if got := UsageReport(records); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
	if got := UsageReport(nil); got != "Token usage: no API requests in this conversation yet." {
		t.Errorf("empty: %q", got)
	}
	// Counts nobody reported stay unknown instead of 0.
	if got := UsageReport(records[1:2]); strings.Contains(got, "Total") || strings.Contains(got, "cached") ||
		!strings.Contains(got, "Output tokens: 0\n") {
		t.Errorf("partial counts:\n%s", got)
	}
	if got := UsageReport([]storage.UsageRecord{{Status: "completed"}}); !strings.Contains(got, "Token usage: not reported") ||
		strings.Contains(got, "Latest request input") {
		t.Errorf("no counts:\n%s", got)
	}
	for n, want := range map[int]string{0: "0", 999: "999", 1000: "1,000", 1234567: "1,234,567", -1500: "-1,500"} {
		if got := formatCount(n); got != want {
			t.Errorf("formatCount(%d) = %q", n, got)
		}
	}
}
