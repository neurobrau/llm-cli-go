package tui

import (
	"strings"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// panelValue is the panel row under heading, or "" if heading is not
// drawn.
func (h *harness) panelValue(heading string) string {
	rows := h.rows()
	for y, row := range rows {
		r := []rune(row)
		if len(r) < 78 || strings.TrimRight(string(r[60:78]), " ") != heading {
			continue
		}
		if y+1 < len(rows) {
			return strings.TrimRight(string([]rune(rows[y+1])[60:78]), " ")
		}
	}
	return ""
}

func (h *harness) waitCost(want string) {
	h.t.Helper()
	h.waitFor("cost "+want, func(string) bool { return h.panelValue("COST (EST.)") == " "+want })
}

// priced is a completed answer with every count and the tier reported.
func priced(text, tier string, input, cached, writes, output int) llm.StreamCompleted {
	c := completed(text, input, output)
	c.Response.ServiceTier = tier
	c.Response.Usage.CachedInputTokens = intp(cached)
	c.Response.Usage.CacheWriteTokens = intp(writes)
	return c
}

// saveConversation stores a conversation with one usage record.
func saveConversation(t *testing.T, dir, id, provider, model string, at time.Time, r storage.UsageRecord) {
	t.Helper()
	r.At, r.Provider, r.Model, r.Status = at, provider, model, storage.StatusCompleted
	c := storage.Conversation{ID: id, CreatedAt: at, UpdatedAt: at, Provider: provider, Model: model,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: model + " question"}, {Role: llm.RoleAssistant, Content: model + " answer"}},
		Requests: []storage.UsageRecord{r}}
	if err := storage.NewHistoryStore(dir).Save(c); err != nil {
		t.Fatal(err)
	}
}

func TestCostFollowsTheConversation(t *testing.T) {
	at := time.Date(2026, 10, 22, 5, 54, 0, 0, time.UTC)
	prepare := func(dir string) {
		// The reference example: 258.5 millionths.
		saveConversation(t, dir, "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f", "openai", "gpt-6-luna", at,
			storage.UsageRecord{ServiceTier: "default", Usage: llm.Usage{InputTokens: intp(2_500), CachedInputTokens: intp(1_600),
				CacheWriteTokens: intp(100), OutputTokens: intp(300), ReasoningTokens: intp(50), TotalTokens: intp(2_800)}})
		// gpt-5.1 bills unreported writes as input: still exact. 1,000 × 1.25 + 10 × 10.
		saveConversation(t, dir, "3f2a1b0c-8d7e-4f6a-9b5c-4d3e2f1a0b9c", "openai", "gpt-5.1", at.Add(-time.Hour),
			storage.UsageRecord{ServiceTier: "default", Usage: llm.Usage{InputTokens: intp(1_000), CachedInputTokens: intp(0),
				OutputTokens: intp(10)}})
	}
	openai := &fakeProvider{name: "openai", defaultModel: "gpt-6-luna",
		models: []llm.Model{{ID: "gpt-6-luna"}, {ID: "gpt-5.1"}},
		turns: []turnScript{
			events(completed("Unqualified", 1_000, 100)), // no tier or cache counts: ~
			events(deltas("Cut")[0], llm.StreamIncomplete{Reason: "the max output token limit was reached", PartialText: "Cut",
				Usage: &llm.Usage{InputTokens: intp(2_000), OutputTokens: intp(1_000)}, ServiceTier: "flex"}),
			events(llm.StreamFailed{Err: &llm.Error{Message: "Server error."}}),
			events(priced("Tiered", "fast", 10_000, 0, 0, 1_000)),
		}}
	ollama := &fakeProvider{name: "ollama", defaultModel: "qwen3.5:4b", models: ollamaModels("qwen3.5:4b"),
		turns: []turnScript{events(completed("Local", 30, 5))}}
	h := start(t, openai, setup{streamed: true, model: "gpt-6-luna", reasoning: "none", prepare: prepare,
		others: map[string]*fakeProvider{"ollama": ollama}})

	// Restored history is priced as it is loaded.
	h.waitCost("$0.000259")
	h.submit("Q1")
	h.waitText("AI > Unqualified")
	h.waitCost("~$0.000409") // + 1,000 × 0.10 + 100 × 0.50
	h.submit("Q2")
	h.waitText("the max output token limit was reached")
	h.waitCost("~$0.000759") // the incomplete answer was billed: + (2,000 × 0.10 + 1,000 × 0.50) × 0.5
	h.submit("Q3")
	h.waitText("Server error.")
	h.waitStatus("ERROR")
	if got := h.panelValue("COST (EST.)"); got != " ~$0.000759" {
		t.Errorf("a failed request changed the cost: %q", got)
	}

	h.submit("/usage")
	h.waitText("Conversation usage (3 request(s), as reported by the API):")
	for _, want := range []string{"Input tokens: 5,500", "of which cached: 1,600", "Output tokens: 1,400",
		"Latest request input: 1,000 tokens", "Includes 1 incomplete answer(s)",
		"Estimated cost: approximately $0.000759 (ESTIMATED, not a billed amount; pricing 2026-10-22;",
		"Standard tier assumed (tier not reported)"} {
		h.waitText(want)
	}

	h.key(tcell.KeyF2)
	h.waitText("SELECT OPENAI MODEL")
	h.key(tcell.KeyDown)
	h.key(tcell.KeyEnter)
	h.waitText("Now using OpenAI / gpt-5.1. Restored its conversation (2 messages).")
	h.waitCost("$0.001350")
	h.submit("Q4")
	h.waitText("AI > Tiered")
	h.waitStatus("DONE")
	if got := h.panelValue("COST (EST.)"); got != " >=$0.001350" { // gpt-5.1 has no fast tier
		t.Errorf("unpriceable tier: %q", got)
	}

	h.key(tcell.KeyF3)
	h.waitText("SELECT PROVIDER")
	h.key(tcell.KeyEnd)
	h.key(tcell.KeyEnter)
	h.waitText("Now using Ollama / qwen3.5:4b. Started a new conversation; the previous one stays saved.")
	h.waitCost("-")
	h.submit("Q5")
	h.waitText("AI > Local")
	h.waitCost("$0 (local)")
	h.submit("/usage")
	h.waitText("API cost: $0 (local Ollama inference)")

	h.key(tcell.KeyF3)
	h.waitText("SELECT PROVIDER")
	h.key(tcell.KeyHome)
	h.key(tcell.KeyEnter)
	h.waitText("Now using OpenAI / gpt-6-luna. Restored its conversation (4 messages).")
	h.waitCost("~$0.000759") // its own records only

	h.submit("/reset")
	h.waitText("Started a new conversation. The previous one stays saved.")
	h.waitCost("-")
	h.submit("/usage")
	h.waitText("Token usage: no API requests in this conversation yet.")
}

func TestCostAfterCleanup(t *testing.T) {
	fake := &fakeProvider{name: "openai", turns: []turnScript{events(priced("A", "default", 2_500, 1_600, 100, 300))}}
	h := start(t, fake, setup{streamed: true, model: "gpt-6-luna-2026-10-22", reasoning: "none"})
	h.waitCost("-")
	h.submit("Q")
	h.waitText("AI > A")
	h.waitCost("$0.000259") // dated snapshots use the base model's prices
	h.submit("/cleanup")
	h.waitStatus("WARNING | Type yes")
	h.submit("yes")
	h.waitText("Deleted 1 saved conversation(s). Started a new conversation.")
	h.waitCost("-")
}

func TestUnpricedModels(t *testing.T) {
	for _, tt := range []struct {
		provider, model, want string
	}{
		{"openai", "gpt-6-luna-mini", "unavailable"},
		{"ollama", "glm-4.7:cloud", "unavailable"}, // never free
		{"ollama", "llama3.2:3b", "$0 (local)"},
	} {
		t.Run(tt.model, func(t *testing.T) {
			fake := &fakeProvider{name: tt.provider, turns: []turnScript{events(completed("A", 10, 10))}}
			h := start(t, fake, setup{streamed: true, model: tt.model})
			h.submit("Q")
			h.waitText("AI > A")
			h.waitCost(tt.want)
		})
	}
}

func TestCostOnSmallScreens(t *testing.T) {
	// 33.6 million output tokens of gpt-5.1 cost $336: the widest common
	// form still fits the 18 panel columns.
	fake := &fakeProvider{name: "openai", turns: []turnScript{events(completed("A", 10_000_000, 33_600_000))}}
	h := start(t, fake, setup{streamed: true, model: "gpt-5.1", reasoning: "none"})
	h.submit("Q")
	h.waitText("AI > A")
	h.waitCost("~$348.5000") // 10M × 1.25 + 33.6M × 10, tier not reported
	if got := h.panelValue("TOKENS"); got != " in 10M out 33.6M" {
		t.Errorf("tokens %q", got)
	}
	if !strings.Contains(h.text(), " context 10M") {
		t.Errorf("context:\n%s", h.text())
	}

	// Too few rows for the cost: it is not drawn, and nothing spills
	// into the status bar.
	h.resize(80, 21)
	h.waitFor("short screen", func(s string) bool { return !strings.Contains(s, "COST") })
	if !strings.Contains(h.text(), " context 10M") {
		t.Errorf("context on a short screen:\n%s", h.text())
	}
	if got := h.row(15); !strings.HasPrefix(got, "+-[ Status ]") {
		t.Errorf("status separator row %q", got)
	}
	// Narrow: no panel at all, and the conversation keeps its width.
	h.resize(63, 25)
	h.waitFor("narrow screen", func(s string) bool { return !strings.Contains(s, "COST") })
	h.resize(80, 25)
	h.waitCost("~$348.5000")
}
