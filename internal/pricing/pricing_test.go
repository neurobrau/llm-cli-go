package pricing

import (
	"math"
	"math/big"
	"reflect"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

func count(n int) *int { return &n }

// openai is an OpenAI record; nil counts are not reported.
func openai(model, tier string, input, cached, writes, output *int) storage.UsageRecord {
	return storage.UsageRecord{Provider: "openai", Model: model, Status: storage.StatusCompleted, ServiceTier: tier,
		Usage: llm.Usage{InputTokens: input, CachedInputTokens: cached, CacheWriteTokens: writes, OutputTokens: output}}
}

// known is an OpenAI record with every count reported.
func known(model, tier string, input, cached, writes, output int) storage.UsageRecord {
	return openai(model, tier, count(input), count(cached), count(writes), count(output))
}

func ollama(model string) storage.UsageRecord {
	return storage.UsageRecord{Provider: "ollama", Model: model, Status: storage.StatusCompleted,
		Usage: llm.Usage{InputTokens: count(10), OutputTokens: count(5)}}
}

// check compares an estimate with the wanted status, exact amount ("" for
// none) and notes.
func check(t *testing.T, name string, got Estimate, status Status, amount string, notes ...string) {
	t.Helper()
	if got.Status != status || !reflect.DeepEqual(got.Notes, notes) {
		t.Errorf("%s: %s %q, want %s %q", name, got.Status, got.Notes, status, notes)
	}
	switch {
	case amount == "" && got.Amount != nil:
		t.Errorf("%s: amount %s, want none", name, got.Amount.RatString())
	case amount != "" && (got.Amount == nil || got.Amount.Cmp(usd(amount)) != 0):
		t.Errorf("%s: amount %v, want %s", name, got.Amount, amount)
	}
}

func TestCatalogRates(t *testing.T) {
	tests := []struct {
		name                          string
		model                         string
		input, cached, writes, output int
		want                          string
	}{
		// gpt-6-luna Standard: 200K tokens are 0.2 of the per-1M price.
		{"luna input", "gpt-6-luna", 200_000, 0, 0, 0, "0.02"},
		{"luna cached input", "gpt-6-luna", 200_000, 200_000, 0, 0, "0.002"},
		{"luna cache write", "gpt-6-luna", 200_000, 0, 200_000, 0, "0.025"},
		{"luna output", "gpt-6-luna", 0, 0, 0, 1_000_000, "0.50"},
		// Long context: the whole request at the higher rates.
		{"luna long input", "gpt-6-luna", 1_000_000, 0, 0, 0, "0.20"},
		{"luna long cached input", "gpt-6-luna", 1_000_000, 1_000_000, 0, 0, "0.02"},
		{"luna long cache write", "gpt-6-luna", 1_000_000, 0, 1_000_000, 0, "0.25"},
		{"luna long output", "gpt-6-luna", 300_000, 0, 0, 1_000_000, "0.81"}, // 0.06 + 0.75
		// gpt-5.1 has no long-context rates and bills writes as input.
		{"5.1 input", "gpt-5.1", 1_000_000, 0, 0, 0, "1.25"},
		{"5.1 cached input", "gpt-5.1", 1_000_000, 1_000_000, 0, 0, "0.125"},
		{"5.1 cache write", "gpt-5.1", 1_000_000, 0, 1_000_000, 0, "1.25"},
		{"5.1 output", "gpt-5.1", 0, 0, 0, 1_000_000, "10"},
		{"5.1 mixed", "gpt-5.1", 1_000, 400, 100, 50, "0.00130"}, // 500*1.25 + 400*0.125 + 100*1.25 + 50*10 = 1300
		{"zero tokens", "gpt-6-luna", 0, 0, 0, 0, "0"},
	}
	for _, tt := range tests {
		got := Request(known(tt.model, "default", tt.input, tt.cached, tt.writes, tt.output))
		check(t, tt.name, got, Estimated, tt.want)
	}
}

func TestReferenceExample(t *testing.T) {
	// 800 × 0.10 + 1,600 × 0.01 + 100 × 0.125 + 300 × 0.50 = 258.5 millionths.
	r := known("gpt-6-luna", "default", 2_500, 1_600, 100, 300)
	r.Usage.ReasoningTokens = count(50) // part of the output: not charged again
	got := Request(r)
	check(t, "reference", got, Estimated, "0.0002585")
	if s := FormatMoney(got.Amount); s != "$0.000259" {
		t.Errorf("display %s", s)
	}
	if s := Describe(Total([]storage.UsageRecord{r})); s !=
		"Estimated cost: $0.000259 (ESTIMATED, not a billed amount; pricing 2026-10-22)" {
		t.Errorf("describe %q", s)
	}
}

func TestLongContextThreshold(t *testing.T) {
	check(t, "at the threshold", Request(known("gpt-6-luna", "default", 272_000, 0, 0, 0)), Estimated, "0.0272")
	check(t, "above it", Request(known("gpt-6-luna", "default", 272_001, 0, 0, 0)), Estimated, "0.0544002")
	// The threshold is about input; much output stays Standard.
	check(t, "output", Request(known("gpt-6-luna", "default", 100, 0, 0, 300_000)), Estimated, "0.15001")
}

func TestServiceTiers(t *testing.T) {
	const assumed = "Standard tier assumed (tier not reported)"
	tests := []struct {
		model, tier string
		status      Status
		amount      string
		notes       []string
	}{
		{"gpt-6-luna", "default", Estimated, "0.02", nil},
		{"gpt-6-luna", "flex", Estimated, "0.01", nil},
		{"gpt-6-luna", "batch", Estimated, "0.01", nil},
		{"gpt-6-luna", "fast", Estimated, "0.04", nil},
		{"gpt-6-luna", "priority", Estimated, "0.04", nil},
		{"gpt-6-luna", "", Qualified, "0.02", []string{assumed}},
		{"gpt-6-luna", "auto", Qualified, "0.02", []string{assumed}},
		{"gpt-6-luna", "scale", Unavailable, "", []string{"no pricing for the scale tier of gpt-6-luna"}},
		{"gpt-6-luna", "Default", Unavailable, "", []string{"no pricing for the Default tier of gpt-6-luna"}},
		{"gpt-5.1", "default", Estimated, "0.25", nil},
		{"gpt-5.1", "", Qualified, "0.25", []string{assumed}},
		{"gpt-5.1", "flex", Unavailable, "", []string{"no pricing for the flex tier of gpt-5.1"}},
		{"gpt-5.1", "priority", Unavailable, "", []string{"no pricing for the priority tier of gpt-5.1"}},
	}
	for _, tt := range tests {
		check(t, tt.model+" "+tt.tier, Request(known(tt.model, tt.tier, 200_000, 0, 0, 0)), tt.status, tt.amount, tt.notes...)
	}
	// A stored `"service_tier": ""` was reported, as an unknown tier; only an
	// absent (or null) tier assumes Standard.
	empty := known("gpt-6-luna", "", 200_000, 0, 0, 0)
	empty.EmptyServiceTier = true
	check(t, "empty tier", Request(empty), Unavailable, "", "no pricing for the  tier of gpt-6-luna")
}

func TestModelIDs(t *testing.T) {
	for model, want := range map[string]string{
		"gpt-6-luna":            "0.02",
		"gpt-6-luna-2026-10-22": "0.02", // dated snapshots use the base prices
		"gpt-5.1-2025-11-13":    "0.25",
	} {
		check(t, model, Request(known(model, "default", 200_000, 0, 0, 0)), Estimated, want)
	}
	// No prefix matching, no case folding, only one well-formed date suffix.
	for _, model := range []string{"gpt-6-luna-mini", "gpt-6", "gpt-5", "gpt-5.1-codex", "GPT-6-LUNA",
		"gpt-6-luna-20261022", "gpt-6-luna-2026-10", "gpt-6-luna-2026-10-22-2026-10-22", "gpt-6-luna:latest", ""} {
		check(t, model, Request(known(model, "default", 200_000, 0, 0, 0)), Unavailable, "", "no pricing data for "+model)
	}
}

func TestMissingAndInconsistentCounts(t *testing.T) {
	const (
		uncached = "cached tokens not reported (all input priced as uncached)"
		asInput  = "cache writes not reported (priced as ordinary input)"
	)
	tests := []struct {
		name   string
		record storage.UsageRecord
		status Status
		amount string
		notes  []string
	}{
		{"cached unknown", openai("gpt-6-luna", "default", count(1000), nil, count(0), count(0)),
			Qualified, "0.0001", []string{uncached}},
		{"writes unknown", openai("gpt-6-luna", "default", count(1000), count(0), nil, count(0)),
			Qualified, "0.0001", []string{asInput}},
		{"both unknown, tier unknown", openai("gpt-6-luna", "", count(1000), nil, nil, count(0)), Qualified, "0.0001",
			[]string{"Standard tier assumed (tier not reported)", uncached, asInput}},
		// gpt-5.1 has no write price: an unreported count changes nothing.
		{"writes unknown without a write price", openai("gpt-5.1", "default", count(1000), count(0), nil, count(0)),
			Estimated, "0.00125", nil},
		{"zero is not unknown", known("gpt-6-luna", "default", 1000, 0, 0, 0), Estimated, "0.0001", nil},
		{"input unknown", openai("gpt-6-luna", "default", nil, count(0), count(0), count(10)),
			Unavailable, "", []string{"token usage was not reported"}},
		{"output unknown", openai("gpt-6-luna", "default", count(10), count(0), count(0), nil),
			Unavailable, "", []string{"token usage was not reported"}},
		{"cached + writes > input", known("gpt-6-luna", "default", 100, 60, 41, 0),
			Unavailable, "", []string{"inconsistent usage (cached + writes > input)"}},
		{"cached + writes = input", known("gpt-6-luna", "default", 100, 60, 40, 0), Estimated, "0.0000056", nil},
		{"cached > input, writes unknown", openai("gpt-6-luna", "default", count(10), count(11), nil, count(0)),
			Unavailable, "", []string{"inconsistent usage (cached + writes > input)"}},
		// Large counts cannot overflow.
		{"huge counts", known("gpt-6-luna", "default", math.MaxInt, math.MaxInt, math.MaxInt, 0),
			Unavailable, "", []string{"inconsistent usage (cached + writes > input)"}},
	}
	for _, tt := range tests {
		check(t, tt.name, Request(tt.record), tt.status, tt.amount, tt.notes...)
	}
	huge := Request(known("gpt-5.1", "default", math.MaxInt, 0, 0, math.MaxInt))
	want := new(big.Rat).SetInt64(math.MaxInt)
	want.Mul(want, usd("11.25")).Quo(want, usd("1000000"))
	if huge.Amount == nil || huge.Amount.Cmp(want) != 0 {
		t.Errorf("huge amount %v", huge.Amount)
	}
}

func TestProviders(t *testing.T) {
	check(t, "local", Request(ollama("qwen3.5:4b")), FreeLocal, "0")
	check(t, "local without tag", Request(ollama("llama3.2")), FreeLocal, "0")
	check(t, "cloud tag", Request(ollama("glm-4.7:cloud")), Unavailable, "", "glm-4.7:cloud is an Ollama cloud model")
	check(t, "cloud suffix", Request(ollama("gemma4:31b-cloud")), Unavailable, "", "gemma4:31b-cloud is an Ollama cloud model")
	check(t, "cloud name", Request(ollama("cloud")), Unavailable, "", "cloud is an Ollama cloud model")
	check(t, "not a cloud tag", Request(ollama("cloudy:4b")), FreeLocal, "0")
	other := known("gpt-6-luna", "default", 10, 0, 0, 10)
	other.Provider = "anthropic"
	check(t, "other provider", Request(other), Unavailable, "", "no pricing for provider anthropic")
	// An OpenAI model name is priced only for the OpenAI provider.
	other.Provider = "ollama"
	check(t, "OpenAI name on Ollama", Request(other), FreeLocal, "0")
}

func TestTotal(t *testing.T) {
	if Total(nil) != nil {
		t.Error("no requests: want nil")
	}
	luna := known("gpt-6-luna", "default", 2_500, 1_600, 100, 300) // 0.0002585
	qualified := openai("gpt-6-luna", "", count(1000), count(0), count(0), count(0))
	unknown := known("gpt-9", "default", 1, 0, 0, 1)
	cloud := ollama("glm-4.7:cloud")
	local := ollama("qwen3.5:4b")
	all := func(records ...storage.UsageRecord) []storage.UsageRecord { return records }
	tests := []struct {
		name    string
		records []storage.UsageRecord
		status  Status
		amount  string
		notes   []string
	}{
		{"all local", all(local, local), FreeLocal, "0", nil},
		{"one estimated", all(luna), Estimated, "0.0002585", nil},
		{"summed exactly", all(luna, luna, luna), Estimated, "0.0007755", nil},
		{"local and OpenAI", all(local, luna), Estimated, "0.0002585", nil},
		{"qualified", all(luna, qualified), Qualified, "0.0003585", []string{"Standard tier assumed (tier not reported)"}},
		{"partial", all(luna, unknown, qualified, unknown), Partial, "0.0003585",
			[]string{"no pricing data for gpt-9", "Standard tier assumed (tier not reported)"}},
		{"none priceable", all(unknown, cloud, unknown), Unavailable, "",
			[]string{"no pricing data for gpt-9", "glm-4.7:cloud is an Ollama cloud model"}},
		// Local inference is known to cost $0, so a cloud request makes the total a lower bound.
		{"local and cloud", all(local, cloud), Partial, "0", []string{"glm-4.7:cloud is an Ollama cloud model"}},
	}
	for _, tt := range tests {
		got := Total(tt.records)
		if got == nil {
			t.Errorf("%s: nil", tt.name)
			continue
		}
		check(t, tt.name, *got, tt.status, tt.amount, tt.notes...)
	}
}

func TestFormatMoney(t *testing.T) {
	for amount, want := range map[string]string{
		"0":            "$0",
		"0.0000005":    "$0.000001", // half up
		"0.00000049":   "$0.000000",
		"0.0000004999": "$0.000000",
		"0.0002585":    "$0.000259",
		"0.0002584999": "$0.000258",
		"0.00995":      "$0.009950",
		"0.0099994999": "$0.009999",
		"0.0099995":    "$0.010000", // below one cent before rounding: 6 places
		"0.01":         "$0.0100",
		"0.01234999":   "$0.0123",
		"0.01235":      "$0.0124",
		"1":            "$1.0000",
		"1234.56785":   "$1234.5679",
		"12345678.9":   "$12345678.9000",
	} {
		if got := FormatMoney(usd(amount)); got != want {
			t.Errorf("FormatMoney(%s) = %s, want %s", amount, got, want)
		}
	}
	// No rounding before summation: each request rounds to $0.000000, the sum does not.
	tiny := known("gpt-6-luna", "default", 4, 0, 0, 0) // 0.0000004
	if got := Short(Total([]storage.UsageRecord{tiny})); got != "$0.000000" {
		t.Errorf("one tiny request: %s", got)
	}
	if got := Short(Total([]storage.UsageRecord{tiny, tiny})); got != "$0.000001" {
		t.Errorf("two tiny requests: %s", got)
	}
}

func TestShortAndDescribe(t *testing.T) {
	luna := known("gpt-6-luna", "default", 2_500, 1_600, 100, 300)
	qualified := openai("gpt-6-luna", "auto", count(1000), nil, count(0), count(0))
	unknown := known("gpt-9", "default", 1, 0, 0, 1)
	tests := []struct {
		name     string
		records  []storage.UsageRecord
		short    string
		describe string
	}{
		{"none", nil, "-", "Estimated cost: - (no requests yet)"},
		{"local", []storage.UsageRecord{ollama("qwen3.5:4b")}, "$0 (local)", "API cost: $0 (local Ollama inference)"},
		{"estimated", []storage.UsageRecord{luna}, "$0.000259",
			"Estimated cost: $0.000259 (ESTIMATED, not a billed amount; pricing 2026-10-22)"},
		{"qualified", []storage.UsageRecord{luna, qualified}, "~$0.000359",
			"Estimated cost: approximately $0.000359 (ESTIMATED, not a billed amount; pricing 2026-10-22; " +
				"Standard tier assumed (tier not reported); cached tokens not reported (all input priced as uncached))"},
		{"partial", []storage.UsageRecord{luna, unknown}, ">=$0.000259",
			"Estimated cost: at least $0.000259 (ESTIMATED, not a billed amount; pricing 2026-10-22; no pricing data for gpt-9)"},
		{"unavailable", []storage.UsageRecord{unknown, ollama("x:cloud")}, "unavailable",
			"Estimated cost: unavailable (no pricing data for gpt-9; x:cloud is an Ollama cloud model)"},
		{"large", []storage.UsageRecord{known("gpt-5.1", "fast", 0, 0, 0, 0), known("gpt-5.1", "default", 0, 0, 0, 1_234_567_891)},
			">=$12345.6789", "Estimated cost: at least $12345.6789 (ESTIMATED, not a billed amount; pricing 2026-10-22; " +
				"no pricing for the fast tier of gpt-5.1)"},
	}
	for _, tt := range tests {
		total := Total(tt.records)
		if got := Short(total); got != tt.short {
			t.Errorf("%s: Short = %q, want %q", tt.name, got, tt.short)
		}
		if got := Describe(total); got != tt.describe {
			t.Errorf("%s: Describe =\n%q\nwant\n%q", tt.name, got, tt.describe)
		}
	}
}

func TestCatalogIsTheReferenceSnapshot(t *testing.T) {
	if Version != "2026-10-22" || len(catalog) != 2 || len(Sources) != 4 {
		t.Errorf("catalog %s with %d models and %d sources", Version, len(catalog), len(Sources))
	}
	for model, tiers := range map[string][]string{
		"gpt-6-luna": {"batch", "default", "fast", "flex", "priority"},
		"gpt-5.1":    {"default"},
	} {
		var got []string
		for _, tier := range []string{"batch", "default", "fast", "flex", "priority", "scale", "auto"} {
			if catalog[model].tiers[tier] != nil {
				got = append(got, tier)
			}
		}
		if !reflect.DeepEqual(got, tiers) || len(catalog[model].tiers) != len(tiers) {
			t.Errorf("%s tiers %v", model, got)
		}
	}
}
