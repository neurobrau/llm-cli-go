package chat

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/pricing"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// TestPythonReadsStreamedConversation is an optional development check:
// a streamed chat with completed, incomplete and failed answers is saved
// by Go in a temp directory, then loaded by the Python 0.3.0 readers,
// which must report no warnings and write it back byte for byte. It runs
// only with
//
//	NEUROBRAU_PYTHON_SRC=$PWD/../llm-cli-python/src go test -run PythonReads ./internal/chat
func TestPythonReadsStreamedConversation(t *testing.T) {
	src := os.Getenv("NEUROBRAU_PYTHON_SRC")
	if src == "" {
		t.Skip("set NEUROBRAU_PYTHON_SRC to run the Python parity check")
	}
	h := streamHarness(t, true,
		events(llm.TextDelta{Text: "Grüße 🎉\n"}, completedWith("Grüße 🎉\n", 10, 3)),
		events(llm.TextDelta{Text: "Part"}, llm.StreamIncomplete{Reason: "r", PartialText: "Part",
			Usage: &llm.Usage{InputTokens: intp(7), OutputTokens: intp(50)}, ServiceTier: "flex"}),
		events(llm.StreamFailed{Err: &llm.Error{Message: "x"}}),
		events(llm.TextDelta{Text: "B"}, completedWith("B", 12, 1)),
	)
	h.run(context.Background(), "Q1\nQ2\nQ3\nQ4\n")
	dir := filepath.Dir(filepath.Dir(h.store.Dir())) // <dir>/neurobrau/conversations

	cmd := exec.Command("python3", "-B", "../../scripts/python_storage_parity.py", "check", src, dir)
	cmd.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("python: %v", err)
	}
	var result struct {
		Conversations map[string]string `json:"conversations"`
		Warnings      []string          `json:"warnings"`
	}
	if err := json.Unmarshal(out, &result); err != nil {
		t.Fatalf("python output: %v\n%s", err, out)
	}
	if len(result.Warnings) != 0 || len(result.Conversations) != 1 {
		t.Fatalf("Python result %+v", result)
	}
	for name, text := range result.Conversations {
		data, _ := os.ReadFile(filepath.Join(h.store.Dir(), name))
		if text != string(data) {
			t.Errorf("%s: Python writes back\n%s\nGo wrote\n%s", name, text, data)
		}
	}
}

// pricingCases are usage-record lists in the conversation-file format:
// every combination of catalog model, tier and reported counts as single
// requests, plus mixed, legacy and malformed conversations.
func pricingCases() map[string][]any {
	cases := map[string][]any{}
	record := func(provider, model, tier, status string, counts map[string]any) map[string]any {
		r := map[string]any{"at": "2026-10-22T05:54:29.609Z", "provider": provider, "model": model}
		if status != "" {
			r["status"] = status
		}
		if tier != "" {
			r["service_tier"] = tier
		}
		for k, v := range counts {
			r[k] = v
		}
		return r
	}
	n := 0
	for _, model := range []string{"gpt-6-luna", "gpt-6-luna-2026-10-22", "gpt-5.1", "gpt-5.1-2025-11-13", "gpt-6-luna-mini", "gpt-5"} {
		for _, tier := range []string{"", "default", "auto", "flex", "batch", "fast", "priority", "scale"} {
			for _, input := range []any{nil, 0, 2_500, 272_000, 272_001, 1_000_003} {
				for _, cached := range []any{nil, 0, 1_600} {
					for _, writes := range []any{nil, 0, 100, 1_000} {
						for _, output := range []any{nil, 0, 300, 7_777_777} {
							counts := map[string]any{"reasoning_tokens": 50}
							for k, v := range map[string]any{"input_tokens": input, "cached_input_tokens": cached,
								"cache_write_tokens": writes, "output_tokens": output} {
								if v != nil {
									counts[k] = v
								}
							}
							n++
							cases[fmt.Sprintf("single %d", n)] = []any{record("openai", model, tier, "completed", counts)}
						}
					}
				}
			}
		}
	}
	usage := func(in, out int) map[string]any {
		return map[string]any{"input_tokens": in, "output_tokens": out, "total_tokens": in + out}
	}
	full := map[string]any{"input_tokens": 2_500, "cached_input_tokens": 1_600, "cache_write_tokens": 100,
		"output_tokens": 300, "reasoning_tokens": 50, "total_tokens": 2_800}
	luna := record("openai", "gpt-6-luna", "default", "completed", full)
	cases["empty"] = []any{}
	cases["reference example"] = []any{luna}
	cases["local"] = []any{record("ollama", "qwen3.5:4b", "", "completed", usage(12, 5)),
		record("ollama", "qwen3.5:4b", "", "incomplete", usage(30, 0))}
	cases["cloud"] = []any{record("ollama", "glm-4.7:cloud", "", "completed", usage(1, 1)),
		record("ollama", "gemma4:31b-cloud", "", "completed", usage(1, 1))}
	cases["local and cloud"] = []any{record("ollama", "qwen3.5:4b", "", "completed", usage(1, 1)),
		record("ollama", "glm-4.7:cloud", "", "completed", usage(1, 1))}
	cases["mixed providers"] = []any{record("ollama", "qwen3.5:4b", "", "completed", usage(10, 5)), luna,
		record("anthropic", "claude", "", "completed", usage(1, 1)),
		record("openai", "gpt-6-luna", "", "incomplete", usage(1_000, 4_000)),
		record("openai", "gpt-9", "default", "completed", usage(1, 1))}
	cases["unknown provider only"] = []any{record("anthropic", "claude", "", "completed", usage(1, 1)),
		record("other", "x", "", "completed", usage(1, 1))}
	cases["incomplete last"] = []any{luna, record("openai", "gpt-6-luna", "flex", "incomplete", usage(9_999, 1))}
	cases["only incomplete"] = []any{record("openai", "gpt-5.1", "default", "incomplete", usage(100, 200))}
	cases["legacy record without status"] = []any{record("openai", "gpt-5.1", "default", "", usage(1_000, 10))}
	emptyTier := record("openai", "gpt-6-luna", "", "completed", usage(1_000, 10))
	emptyTier["service_tier"] = ""
	nullTier := record("openai", "gpt-6-luna", "", "completed", usage(1_000, 10))
	nullTier["service_tier"] = nil
	cases["empty tier"] = []any{emptyTier}
	cases["null tier"] = []any{nullTier}
	cases["empty and absent tiers"] = []any{emptyTier, record("openai", "gpt-6-luna", "", "completed", usage(1_000, 10)), luna}
	cases["empty tier on gpt-5.1"] = []any{record("openai", "gpt-5.1", "", "completed", usage(5, 5))}
	cases["empty tier on gpt-5.1"][0].(map[string]any)["service_tier"] = ""
	cases["record without counts"] = []any{record("openai", "gpt-6-luna", "default", "completed", nil)}
	cases["tiny amounts summed"] = []any{record("openai", "gpt-6-luna", "default", "completed",
		map[string]any{"input_tokens": 4, "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0}),
		record("openai", "gpt-6-luna", "default", "completed",
			map[string]any{"input_tokens": 4, "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0})}
	cases["malformed records skipped"] = []any{luna, "not a record", 42,
		record("openai", "gpt-6-luna", "default", "completed", map[string]any{"input_tokens": -1}),
		record("openai", "gpt-6-luna", "default", "completed", map[string]any{"output_tokens": true}),
		record("openai", "gpt-6-luna", "default", "completed", map[string]any{"output_tokens": 1.5}),
		record("openai", "gpt-6-luna", "default", "failed", usage(1, 1)),
		map[string]any{"at": "yesterday", "provider": "openai", "model": "gpt-6-luna"},
		map[string]any{"at": "2026-10-22T05:54:29.609Z", "model": "gpt-6-luna"},
		record("openai", "gpt-6-luna", "", "completed", map[string]any{"service_tier": 3}),
		record("openai", "gpt-5.1", "default", "completed", usage(100, 1))}
	return cases
}

// TestPythonPricingParity is an optional development check: the Go and
// the Python 0.3.0 cost estimates, panel cost and /usage report agree on
// the same records, parsed from the same JSON by each implementation's
// readers. The /usage report needs the reference's dependencies, so
// NEUROBRAU_PYTHON names an interpreter that has them (default python3).
// It runs only with
//
//	NEUROBRAU_PYTHON_SRC=$PWD/../llm-cli-python/src \
//	NEUROBRAU_PYTHON=$PWD/../llm-cli-python/.venv/bin/python \
//	go test -run PythonPricing ./internal/chat
func TestPythonPricingParity(t *testing.T) {
	cases := pricingCases()
	out := runPythonScript(t, "python_pricing_parity.py", cases)
	type estimate struct {
		Status string   `json:"status"`
		Amount *string  `json:"amount"`
		Notes  []string `json:"notes"`
	}
	var results map[string]struct {
		Estimates []estimate `json:"estimates"`
		Short     string     `json:"short"`
		Describe  string     `json:"describe"`
		Report    string     `json:"report"`
	}
	if err := json.Unmarshal(out, &results); err != nil {
		t.Fatalf("python output: %v", err)
	}
	if len(results) != len(cases) {
		t.Fatalf("Python returned %d cases, want %d", len(results), len(cases))
	}
	statuses := map[string]int{}
	for name, raw := range cases {
		requests, _ := json.Marshal(raw)
		c, err := storage.DecodeConversation([]byte(`{"schema_version": 1, "id": "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f",
			"created_at": "2026-10-22T05:54:00.123Z", "updated_at": "2026-10-22T05:54:29.609Z",
			"messages": [], "requests": ` + string(requests) + `}`))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want := results[name]
		if len(c.Requests) != len(want.Estimates) {
			t.Errorf("%s: Go keeps %d records, Python %d", name, len(c.Requests), len(want.Estimates))
			continue
		}
		for i, r := range c.Requests {
			got, w := pricing.Request(r), want.Estimates[i]
			statuses[string(got.Status)]++
			var amount *big.Rat
			if w.Amount != nil {
				var ok bool
				if amount, ok = new(big.Rat).SetString(*w.Amount); !ok {
					t.Fatalf("%s: Python amount %q", name, *w.Amount)
				}
			}
			if string(got.Status) != w.Status || !slices.Equal(got.Notes, w.Notes) ||
				(amount == nil) != (got.Amount == nil) || (amount != nil && amount.Cmp(got.Amount) != 0) {
				t.Errorf("%s request %d: Go %s %v %q, Python %+v", name, i, got.Status, got.Amount, got.Notes, w)
			}
		}
		total := pricing.Total(c.Requests)
		if got := pricing.Short(total); got != want.Short {
			t.Errorf("%s: Go panel cost %q, Python %q", name, got, want.Short)
		}
		if got := pricing.Describe(total); got != want.Describe {
			t.Errorf("%s: Go cost line\n%q\nPython\n%q", name, got, want.Describe)
		}
		report := UsageReport(c.Requests)
		if report != want.Report {
			t.Errorf("%s: Go /usage\n%s\nPython\n%s", name, report, want.Report)
		}
	}
	t.Logf("%d cases compared; per-request statuses %v", len(cases), statuses)
}

// runPythonScript runs a parity script of scripts/ on input, written as
// JSON to a temp file, and returns its stdout. It skips the test unless
// NEUROBRAU_PYTHON_SRC is set and an interpreter (NEUROBRAU_PYTHON,
// default python3) is found.
func runPythonScript(t *testing.T, script string, input any) []byte {
	t.Helper()
	src := os.Getenv("NEUROBRAU_PYTHON_SRC")
	if src == "" {
		t.Skip("set NEUROBRAU_PYTHON_SRC to run the Python parity check")
	}
	python := cmp.Or(os.Getenv("NEUROBRAU_PYTHON"), "python3")
	if _, err := exec.LookPath(python); err != nil {
		t.Skipf("no Python interpreter for the parity check: %v", err)
	}
	data, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "cases.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(python, "-B", filepath.Join("..", "..", "scripts", script), src, path)
	cmd.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("python (set NEUROBRAU_PYTHON to an interpreter with the reference's dependencies): %v\n%s", err, stderr.String())
	}
	return out
}

type supportCase struct {
	Values        []string `json:"values"`
	Known         bool     `json:"known"`
	Default       *string  `json:"default"`
	AllowsDefault bool     `json:"allows_default"`
}

type settingsCase struct {
	Support      supportCase    `json:"support"`
	SupportError string         `json:"support_error"`
	Limit        *int           `json:"limit"`
	Prefs        map[string]any `json:"prefs"`
	Overrides    map[string]any `json:"overrides"`
	Strict       bool           `json:"strict"`
}

// TestPythonSettingsParity is an optional development check: Go's
// generation-settings resolution (command line > saved > built-in, with
// the model's support) agrees with Python 0.3.0's
// preferences.resolve_model_settings on every combination of support,
// saved value, override and strictness. It runs like
// TestPythonPricingParity (go test -run PythonSettings ./internal/chat).
func TestPythonSettingsParity(t *testing.T) {
	str := func(s string) *string { return &s }
	supports := []struct {
		support supportCase
		err     string
	}{
		{supportCase{[]string{"none", "low", "medium", "high", "xhigh"}, true, str("none"), true}, ""}, // gpt-6-luna
		{supportCase{[]string{"off", "on"}, true, str("off"), false}, ""},                              // Ollama thinking
		{supportCase{[]string{"low", "medium", "high"}, true, nil, true}, ""},                          // named levels
		{supportCase{[]string{}, true, nil, true}, ""},                                                 // no reasoning
		{supportCase{[]string{}, false, nil, true}, ""},                                                // unknown
		{supportCase{[]string{}, false, nil, true}, "Could not connect to Ollama."},
	}
	var cases []settingsCase
	for _, sup := range supports {
		for _, saved := range []any{nil, "default", "none", "off", "on", "low", "high", "minimal", "xhigh"} {
			for _, savedMax := range []map[string]any{{}, {"max_output_tokens_set": true},
				{"max_output_tokens_set": true, "max_output_tokens": 600}, {"max_output_tokens_set": true, "max_output_tokens": 5000}} {
				for _, flag := range []any{nil, llm.ModelDefault, "none", "off", "low", "high", "minimal", "on", "xhigh"} {
					for _, flagMax := range []any{nil, 100, 5000} {
						for _, limit := range []*int{nil, intp(4096)} {
							for _, strict := range []bool{false, true} {
								prefs := map[string]any{"reasoning": saved}
								for k, v := range savedMax {
									prefs[k] = v
								}
								cases = append(cases, settingsCase{Support: sup.support, SupportError: sup.err, Limit: limit,
									Prefs: prefs, Overrides: map[string]any{"reasoning": flag, "max_output_tokens": flagMax}, Strict: strict})
							}
						}
					}
				}
			}
		}
	}
	out := runPythonScript(t, "python_settings_parity.py", cases)
	var results []struct {
		Error     *string  `json:"error"`
		Reasoning *string  `json:"reasoning"`
		Explicit  bool     `json:"explicit"`
		MaxTokens *int     `json:"max_output_tokens"`
		Notices   []string `json:"notices"`
	}
	if err := json.Unmarshal(out, &results); err != nil || len(results) != len(cases) {
		t.Fatalf("python output: %v, %d results for %d cases", err, len(results), len(cases))
	}
	failures := 0
	for i, c := range cases {
		support := llm.ReasoningSupport{Values: c.Support.Values, Known: c.Support.Known,
			Default: deref(c.Support.Default), NoModelDefault: !c.Support.AllowsDefault}
		if len(support.Values) == 0 {
			support.Values = nil
		}
		p := &fakeProvider{support: support}
		if c.SupportError != "" {
			p = &fakeProvider{supErr: &llm.Error{Kind: llm.KindConnection, Message: c.SupportError}}
			support = llm.ReasoningSupport{}
		}
		if c.Limit != nil {
			p.limit = *c.Limit
		}
		var prefs storage.ModelPreferences
		if r, ok := c.Prefs["reasoning"].(string); ok {
			prefs.Reasoning = r
		}
		prefs.MaxOutputTokensSet = c.Prefs["max_output_tokens_set"] == true
		if n, ok := c.Prefs["max_output_tokens"].(int); ok {
			prefs.MaxOutputTokens = n
		}
		var o Overrides
		if r, ok := c.Overrides["reasoning"].(string); ok {
			o.Reasoning = r
		}
		if n, ok := c.Overrides["max_output_tokens"].(int); ok {
			o.MaxOutputTokens = n
		}
		saved := storage.DefaultSettings().WithModel("openai", "m", prefs)
		resolve := ResolveSwitchedSettings
		if c.Strict {
			resolve = ResolveSettings
		}
		s, notices, err := resolve(context.Background(), p, "m", saved, o)
		w := results[i]
		var got, want string
		if err != nil {
			got = "error: " + err.Error()
		} else {
			got = fmt.Sprintf("reasoning %q explicit %v max %d notices %q", s.EffectiveReasoning(support),
				s.Reasoning.Explicit && s.Reasoning.Effort != "", s.MaxOutputTokens, notices)
		}
		if w.Error != nil {
			want = "error: " + *w.Error
		} else {
			want = fmt.Sprintf("reasoning %q explicit %v max %d notices %q", deref(w.Reasoning),
				w.Explicit && w.Reasoning != nil, deref(w.MaxTokens), w.Notices)
		}
		if got != want {
			if failures++; failures <= 20 {
				t.Errorf("case %+v:\n   Go %s\nPython %s", c, got, want)
			}
		}
	}
	t.Logf("%d cases compared, %d differ", len(cases), failures)
}

// deref is *p, or the zero value for nil.
func deref[T any](p *T) T {
	var zero T
	if p == nil {
		return zero
	}
	return *p
}
