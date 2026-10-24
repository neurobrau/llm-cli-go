// Package pricing estimates API costs from a small, versioned pricing
// catalog (docs/tui-spec.md, section 15).
//
// These are estimates, not billed amounts. Only prices verified on
// OpenAI's official pages are listed (see Sources); anything that cannot
// be priced reliably is reported as unavailable or qualified, never
// guessed. All arithmetic is exact (math/big.Rat), never binary floating
// point; rounding happens only for display.
//
// Billing rules applied (from the official pricing and prompt-caching
// docs):
//   - Cached input and cache-write tokens are parts of the input tokens;
//     ordinary input = input - cached - writes. Each part is billed at its
//     own rate (cache-write rates replace, not add to, the input rate).
//     Models without a cache-write price bill writes as ordinary input.
//   - Reasoning tokens are part of the output tokens and billed as output.
//   - Above a model's long-context threshold (input tokens of the
//     request), the long-context rates apply to the whole request.
//   - Processing tiers scale the Standard rates (Flex/Batch 50%, Fast 2x)
//     for models whose tiers are listed; a tier without a known price is
//     unavailable.
//   - This app never sets prompt_cache_retention, so the default "30m"
//     cache-write rate applies. Regional/data-residency endpoints (+10%)
//     are not used.
package pricing

import (
	"fmt"
	"math/big"
	"regexp"
	"strings"

	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// Version is the date the catalog was verified.
const Version = "2026-10-22"

// Sources are the official pages the catalog was verified on.
var Sources = []string{
	"https://developers.openai.com/api/docs/pricing",
	"https://developers.openai.com/api/docs/models/gpt-6-luna",
	"https://developers.openai.com/api/docs/models/gpt-5.1",
	"https://developers.openai.com/api/docs/guides/prompt-caching",
}

// Status says how far an estimate can be trusted.
type Status string

const (
	Estimated   Status = "estimated"   // all rates and usage known
	Qualified   Status = "qualified"   // computed, but on stated assumptions (see Notes)
	Partial     Status = "partial"     // some requests could not be priced: a lower bound
	Unavailable Status = "unavailable" // no amount
	FreeLocal   Status = "free_local"  // confirmed local inference: no API cost
)

// Estimate is the estimated cost of one request or a conversation. Amount
// is in USD and nil when no amount is known.
type Estimate struct {
	Status Status
	Amount *big.Rat
	Notes  []string
}

// rates are USD per 1M tokens. A nil cacheWrite bills writes as ordinary
// input.
type rates struct {
	input, cachedInput, output, cacheWrite *big.Rat
}

type modelPricing struct {
	standard    rates
	longContext *rates
	threshold   int // input tokens; above it, the long-context rates apply
	tiers       map[string]*big.Rat
}

var (
	perTokens      = big.NewRat(1_000_000, 1) // catalog prices are per 1M tokens
	snapshotSuffix = regexp.MustCompile(`-\d{4}-\d{2}-\d{2}$`)
)

// usd parses an exact decimal price.
func usd(s string) *big.Rat {
	r, ok := new(big.Rat).SetString(s)
	if !ok {
		panic("pricing: bad price " + s)
	}
	return r
}

// catalog was verified on 2026-10-22 on the official pages in Sources.
var catalog = map[string]modelPricing{
	"gpt-6-luna": {
		standard:    rates{input: usd("0.10"), cachedInput: usd("0.01"), output: usd("0.50"), cacheWrite: usd("0.125")},
		longContext: &rates{input: usd("0.20"), cachedInput: usd("0.02"), output: usd("0.75"), cacheWrite: usd("0.25")},
		threshold:   272_000,
		// "priority" was renamed "fast" on 2026-07-30.
		tiers: map[string]*big.Rat{"default": usd("1"), "flex": usd("0.5"), "batch": usd("0.5"), "fast": usd("2"), "priority": usd("2")},
	},
	"gpt-5.1": {
		standard: rates{input: usd("1.25"), cachedInput: usd("0.125"), output: usd("10.00")}, // no cache-write charge
		tiers:    map[string]*big.Rat{"default": usd("1")},
	},
}

// lookup finds a model's prices by exact ID or, for a dated snapshot such
// as gpt-5.1-2025-11-13, by its base model.
func lookup(model string) (modelPricing, bool) {
	if p, ok := catalog[model]; ok {
		return p, true
	}
	p, ok := catalog[snapshotSuffix.ReplaceAllString(model, "")]
	return p, ok
}

// isCloudName reports whether an Ollama model tag marks a cloud model
// (`glm-4.7:cloud`, `gemma4:31b-cloud`); the same rule as the Ollama
// provider's.
func isCloudName(model string) bool {
	tag := model[strings.LastIndex(model, ":")+1:]
	return tag == "cloud" || strings.HasSuffix(tag, "-cloud")
}

func unavailable(note string) Estimate {
	return Estimate{Status: Unavailable, Notes: []string{note}}
}

// Request estimates the cost of one request from its reported usage.
func Request(r storage.UsageRecord) Estimate {
	switch r.Provider {
	case "ollama":
		if isCloudName(r.Model) {
			return unavailable(r.Model + " is an Ollama cloud model")
		}
		return Estimate{Status: FreeLocal, Amount: new(big.Rat)}
	case "openai":
	default:
		return unavailable("no pricing for provider " + r.Provider)
	}
	p, ok := lookup(r.Model)
	if !ok {
		return unavailable("no pricing data for " + r.Model)
	}
	u := r.Usage
	if u.InputTokens == nil || u.OutputTokens == nil {
		return unavailable("token usage was not reported")
	}

	var notes []string
	var multiplier *big.Rat
	switch tier := r.ServiceTier; {
	case (tier == "" && !r.EmptyServiceTier) || tier == "auto":
		multiplier = usd("1")
		notes = append(notes, "Standard tier assumed (tier not reported)")
	case p.tiers[tier] != nil:
		multiplier = p.tiers[tier]
	default:
		return unavailable(fmt.Sprintf("no pricing for the %s tier of %s", tier, r.Model))
	}

	rt := p.standard
	if p.longContext != nil && *u.InputTokens > p.threshold {
		rt = *p.longContext
	}
	var cached, writes int64
	if u.CachedInputTokens == nil {
		notes = append(notes, "cached tokens not reported (all input priced as uncached)")
	} else {
		cached = int64(*u.CachedInputTokens)
	}
	if u.CacheWriteTokens == nil {
		if rt.cacheWrite != nil {
			notes = append(notes, "cache writes not reported (priced as ordinary input)")
		}
	} else {
		writes = int64(*u.CacheWriteTokens)
	}
	// In big.Int, so that no sum of reported counts can overflow.
	ordinary := big.NewInt(int64(*u.InputTokens))
	ordinary.Sub(ordinary, big.NewInt(cached)).Sub(ordinary, big.NewInt(writes))
	if ordinary.Sign() < 0 {
		return unavailable("inconsistent usage (cached + writes > input)")
	}
	writeRate := rt.cacheWrite
	if writeRate == nil {
		writeRate = rt.input
	}
	amount := new(big.Rat)
	for _, part := range []struct {
		tokens *big.Int
		rate   *big.Rat
	}{
		{ordinary, rt.input},
		{big.NewInt(cached), rt.cachedInput},
		{big.NewInt(writes), writeRate},
		{big.NewInt(int64(*u.OutputTokens)), rt.output},
	} {
		amount.Add(amount, new(big.Rat).Mul(new(big.Rat).SetInt(part.tokens), part.rate))
	}
	amount.Mul(amount, multiplier).Quo(amount, perTokens)
	status := Estimated
	if len(notes) > 0 {
		status = Qualified
	}
	return Estimate{Status: status, Amount: amount, Notes: notes}
}

// Total is the combined estimate of a conversation's requests; nil if
// there are none. Notes are deduplicated in first-seen order.
func Total(records []storage.UsageRecord) *Estimate {
	if len(records) == 0 {
		return nil
	}
	statuses := map[Status]bool{}
	seen := map[string]bool{}
	var notes []string
	var amount *big.Rat
	for _, r := range records {
		e := Request(r)
		statuses[e.Status] = true
		for _, note := range e.Notes {
			if !seen[note] {
				seen[note] = true
				notes = append(notes, note)
			}
		}
		if e.Amount != nil {
			if amount == nil {
				amount = new(big.Rat)
			}
			amount.Add(amount, e.Amount)
		}
	}
	switch {
	case len(statuses) == 1 && statuses[FreeLocal]:
		return &Estimate{Status: FreeLocal, Amount: new(big.Rat)}
	case amount == nil:
		return &Estimate{Status: Unavailable, Notes: notes}
	case statuses[Unavailable]:
		return &Estimate{Status: Partial, Amount: amount, Notes: notes}
	case statuses[Qualified]:
		return &Estimate{Status: Qualified, Amount: amount, Notes: notes}
	}
	return &Estimate{Status: Estimated, Amount: amount, Notes: notes}
}

var cent = big.NewRat(1, 100)

// FormatMoney writes a USD amount for display: 6 decimals below one cent,
// otherwise 4, rounded half up (display rounding only); `$0` for zero.
func FormatMoney(amount *big.Rat) string {
	if amount.Sign() == 0 {
		return "$0"
	}
	places := 4
	if amount.Cmp(cent) < 0 {
		places = 6
	}
	scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(places)), nil)
	// Half up, away from zero: floor(|amount| * 10^places + 1/2).
	scaled := new(big.Rat).Abs(amount)
	scaled.Mul(scaled, new(big.Rat).SetInt(scale)).Add(scaled, big.NewRat(1, 2))
	digits := new(big.Int).Quo(scaled.Num(), scaled.Denom()).String()
	if len(digits) <= places {
		digits = strings.Repeat("0", places-len(digits)+1) + digits
	}
	sign := ""
	if amount.Sign() < 0 {
		sign = "-"
	}
	point := len(digits) - places
	return "$" + sign + digits[:point] + "." + digits[point:]
}

// Short is the compact form for the settings panel: `$0.001234`, `~$…`
// qualified, `>=$…` partial, `$0 (local)`, `unavailable`, or `-` without
// requests.
func Short(e *Estimate) string {
	switch {
	case e == nil:
		return "-"
	case e.Status == FreeLocal:
		return "$0 (local)"
	case e.Amount == nil:
		return "unavailable"
	}
	return map[Status]string{Qualified: "~", Partial: ">="}[e.Status] + FormatMoney(e.Amount)
}

// Describe is the cost line of /usage, always labelled as an estimate.
func Describe(e *Estimate) string {
	switch {
	case e == nil:
		return "Estimated cost: - (no requests yet)"
	case e.Status == FreeLocal:
		return "API cost: $0 (local Ollama inference)"
	case e.Amount == nil:
		return "Estimated cost: unavailable (" + strings.Join(e.Notes, "; ") + ")"
	}
	label := map[Status]string{Qualified: "approximately ", Partial: "at least "}[e.Status]
	detail := ""
	if len(e.Notes) > 0 {
		detail = "; " + strings.Join(e.Notes, "; ")
	}
	return fmt.Sprintf("Estimated cost: %s%s (ESTIMATED, not a billed amount; pricing %s%s)",
		label, FormatMoney(e.Amount), Version, detail)
}
