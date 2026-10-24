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

	// Without a choice the support is looked up for the built-in default.
	p = &fakeProvider{supErr: &llm.Error{Kind: llm.KindConnection, Message: "Could not connect to Ollama."}}
	got, notices, err = ResolveSettings(context.Background(), p, "m", storage.DefaultSettings(), Overrides{})
	if err != nil || got.Reasoning != (llm.Reasoning{}) ||
		!reflect.DeepEqual(notices, []string{"Could not check reasoning support for m: Could not connect to Ollama."}) {
		t.Errorf("nothing chosen: %+v, %q, %v", got, notices, err)
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
		{At: at, Provider: "openai", Model: "gpt-6-luna", Status: "completed", ServiceTier: "default",
			Usage: llm.Usage{InputTokens: intp(1000), OutputTokens: intp(200), TotalTokens: intp(1200),
				CachedInputTokens: intp(600), CacheWriteTokens: intp(100), ReasoningTokens: intp(50)}},
		{At: at, Provider: "openai", Model: "gpt-6-luna", Status: "completed", ServiceTier: "default",
			Usage: llm.Usage{InputTokens: intp(1250), OutputTokens: intp(0)}},
		{At: at, Provider: "openai", Model: "gpt-6-luna", Status: "incomplete", ServiceTier: "default",
			Usage: llm.Usage{InputTokens: intp(1300), OutputTokens: intp(4000), ReasoningTokens: intp(0)}},
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
Estimated cost: approximately $0.002404 (ESTIMATED, not a billed amount; pricing 2026-10-22; ` +
		`cached tokens not reported (all input priced as uncached); cache writes not reported (priced as ordinary input))`
	// 148.5 + 125 + 2130 = 2403.5 millionths, rounded half up only for display.
	if got := UsageReport(records); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
	if got := UsageReport(nil); got != "Token usage: no API requests in this conversation yet." {
		t.Errorf("empty: %q", got)
	}
	// Counts nobody reported stay unknown instead of 0.
	if got := UsageReport(records[1:2]); strings.Contains(got, "Total") || strings.Contains(got, "of which cached") ||
		!strings.Contains(got, "Output tokens: 0\n") {
		t.Errorf("partial counts:\n%s", got)
	}
	// A hand-edited record without counts: no count lines, nothing invented.
	if got := UsageReport([]storage.UsageRecord{{Provider: "openai", Model: "gpt-6-luna", Status: "completed"}}); got !=
		"Conversation usage (1 request(s), as reported by the API):\n"+
			"Estimated cost: unavailable (token usage was not reported)" {
		t.Errorf("no counts:\n%s", got)
	}
	for n, want := range map[int]string{0: "0", 999: "999", 1000: "1,000", 1234567: "1,234,567", -1500: "-1,500"} {
		if got := formatCount(n); got != want {
			t.Errorf("formatCount(%d) = %q", n, got)
		}
	}
}

func TestEffectiveReasoning(t *testing.T) {
	support := llm.ReasoningSupport{Known: true, Values: []string{"none", "low"}, Default: "none"}
	tests := []struct {
		settings Settings
		want     string
	}{
		{Settings{}, "none"}, // NeuroBräu's default
		{Settings{Reasoning: llm.Reasoning{Explicit: true}}, ""},                   // model default: no parameter
		{Settings{Reasoning: llm.Reasoning{Effort: "low", Explicit: true}}, "low"}, // chosen
	}
	for _, tt := range tests {
		if got := tt.settings.EffectiveReasoning(support); got != tt.want {
			t.Errorf("%+v: %q, want %q", tt.settings, got, tt.want)
		}
	}
	if got := (Settings{}).EffectiveReasoning(llm.ReasoningSupport{}); got != "" {
		t.Errorf("unknown support: %q", got)
	}
}

// In the fullscreen UI an override the switched-to model does not support
// is a notice; the model's saved value is used instead.
func TestResolveSwitchedSettings(t *testing.T) {
	p := &fakeProvider{support: onOffSupport, limit: 4096}
	saved := storage.DefaultSettings().WithModel("openai", "qwen3.5:4b",
		storage.ModelPreferences{Reasoning: "on", MaxOutputTokensSet: true, MaxOutputTokens: 512})
	o := Overrides{Reasoning: llm.ModelDefault, MaxOutputTokens: 9000}
	if _, _, err := ResolveSettings(context.Background(), p, "qwen3.5:4b", saved, o); err == nil {
		t.Fatal("strict resolution accepted an unsupported override")
	}
	s, notices, err := ResolveSwitchedSettings(context.Background(), p, "qwen3.5:4b", saved, o)
	if err != nil {
		t.Fatal(err)
	}
	if s.Reasoning != (llm.Reasoning{Effort: "on", Explicit: true}) || s.MaxOutputTokens != 512 {
		t.Errorf("settings %+v", s)
	}
	want := []string{
		"qwen3.5:4b has no model-default reasoning choice (qwen3.5:4b supports reasoning: Off, On); using the default.",
		"9000 exceeds the limit of 4096 output tokens for qwen3.5:4b. Using the default.",
	}
	if !reflect.DeepEqual(notices, want) {
		t.Errorf("notices %q", notices)
	}
}

// Malformed records are skipped by storage; a legacy record without a
// status counts as completed. Only the valid records are priced.
func TestUsageReportOfStoredRecords(t *testing.T) {
	c, err := storage.DecodeConversation([]byte(`{"schema_version": 1, "id": "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f",
	  "created_at": "2026-10-22T05:54:00.123Z", "updated_at": "2026-10-22T05:54:29.609Z", "messages": [],
	  "requests": [
	    {"at": "2026-10-22T05:54:29.609Z", "provider": "openai", "model": "gpt-5.1", "service_tier": "default",
	     "input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 10},
	    {"at": "2026-10-22T05:54:29.609Z", "provider": "openai", "model": "gpt-5.1", "input_tokens": -5, "output_tokens": 1},
	    {"at": "not a time", "provider": "openai", "model": "gpt-5.1", "input_tokens": 5, "output_tokens": 1},
	    "garbage",
	    {"at": "2026-10-22T05:55:00.000Z", "provider": "ollama", "model": "glm-4.7:cloud", "status": "incomplete",
	     "input_tokens": 7, "output_tokens": 3}]}`))
	if err != nil {
		t.Fatal(err)
	}
	want := `Conversation usage (2 request(s), as reported by the API):
Input tokens: 1,007
  of which cached: 0
Output tokens: 13
Latest request input: 1,000 tokens (the whole history is sent, and billed, with every request)
Includes 1 incomplete answer(s): billed, but not saved as messages.
Estimated cost: at least $0.001350 (ESTIMATED, not a billed amount; pricing 2026-10-22; glm-4.7:cloud is an Ollama cloud model)`
	if got := UsageReport(c.Requests); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
}
