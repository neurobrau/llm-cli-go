package tui

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

func TestMenuKeys(t *testing.T) {
	items := make([]menuItem, 20)
	for i := range items {
		items[i] = menuItem{label: string(rune('a' + i)), current: i == 3}
	}
	m := newMenu(menuModel, "T", items)
	if m.index != 3 || m.top != 3 {
		t.Fatalf("opens at %d/%d", m.index, m.top)
	}
	steps := []struct {
		key       tcell.Key
		index     int
		action    menuAction
		top       int
		visibleOK bool
	}{
		{tcell.KeyUp, 2, menuNone, 2, true},
		{tcell.KeyPgDn, 6, menuNone, 2, true}, // visible 5: moves by 4
		{tcell.KeyDown, 7, menuNone, 3, true},
		{tcell.KeyEnd, 19, menuNone, 15, true},
		{tcell.KeyDown, 19, menuNone, 15, true}, // clamped
		{tcell.KeyPgUp, 15, menuNone, 15, true},
		{tcell.KeyHome, 0, menuNone, 0, true},
		{tcell.KeyUp, 0, menuNone, 0, true},
		{tcell.KeyRune, 0, menuNone, 0, true}, // ignored
		{tcell.KeyEnter, 0, menuSelect, 0, true},
		{tcell.KeyEsc, 0, menuCancel, 0, true},
		{tcell.KeyCtrlD, 0, menuCancel, 0, true},
	}
	for i, s := range steps {
		if action := m.key(s.key, 5); action != s.action || m.index != s.index || m.top != s.top {
			t.Errorf("step %d (%v): action %v index %d top %d, want %v %d %d", i, s.key, action, m.index, m.top, s.action, s.index, s.top)
		}
	}
}

func TestMenuLine(t *testing.T) {
	item := menuItem{label: "qwen3.5:4b", note: "4.7B Q4_K_M", current: true}
	if got := menuLine(item, 26); got != "* qwen3.5:4b   4.7B Q4_K_M" { // right-aligned
		t.Errorf("%q", got)
	}
	if got := menuLine(item, 14); got != "* qwen3.5:4b  " { // the note does not fit
		t.Errorf("%q", got)
	}
	if got := menuLine(menuItem{label: "a-very-long-model-name"}, 10); got != "  a-very-~" {
		t.Errorf("%q", got)
	}
	if got := menuLine(item, 0); got != "* qwen3.5:4b  4.7B Q4_K_M" {
		t.Errorf("natural %q", got)
	}
}

// styleMenuRows is the F6 menu at 80x25: 39 wide, centered over the
// conversation (columns 0–58) at x 9, rows 7–11.
var styleMenuRows = []string{
	"+-[ ANSWER STYLE ]--------------------+",
	"|   Brief      short, concise answers |",
	"| * Normal           standard answers |",
	"|   Detailed  thorough, with examples |",
	"+-[ 2/3 Enter=OK Esc=cancel ]---------+",
}

func (h *harness) menuAt(x, y int, rows []string) bool {
	all := h.rows()
	for i, want := range rows {
		r := []rune(all[y+i])
		if x+len([]rune(want)) > len(r) || string(r[x:x+len([]rune(want))]) != want {
			return false
		}
	}
	return true
}

func TestStyleMenuLayout(t *testing.T) {
	h := start(t, &fakeProvider{}, setup{})
	h.typeText("draft")
	h.key(tcell.KeyF6)
	h.waitFor("menu", func(string) bool { return h.menuAt(9, 7, styleMenuRows) })
	if _, style := h.cell(10, 9); style != newStyles(256)[roleSelected] {
		t.Error("the current item is not highlighted")
	}
	if _, style := h.cell(10, 8); style != newStyles(256)[roleText] {
		t.Error("other items are highlighted")
	}
	// Keys go to the menu only.
	h.typeText("xyz/")
	h.key(tcell.KeyPgUp)
	h.key(tcell.KeyF2)
	h.key(tcell.KeyEnd)
	h.waitText("3/3 Enter=OK")
	if _, style := h.cell(10, 10); style != newStyles(256)[roleSelected] {
		t.Error("End did not move the highlight")
	}

	// 63 columns: no panel; centered over the whole width.
	h.resize(63, 25)
	h.waitFor("narrow menu", func(string) bool {
		return h.menuAt(12, 7, []string{"+-[ ANSWER STYLE ]--------------------+"})
	})
	// Too small: the warning, then the menu comes back as it was.
	h.resize(30, 10)
	h.waitText("Terminal too small.")
	h.resize(80, 25)
	h.waitText("3/3 Enter=OK")
	h.key(tcell.KeyEsc)
	h.waitFor("closed", func(s string) bool { return !strings.Contains(s, "ANSWER STYLE ]") })
	if !strings.Contains(h.text(), "| YOU > draft ") || strings.Contains(h.text(), "xyz") {
		t.Errorf("draft lost or keys leaked:\n%s", h.text())
	}
	if _, err := os.Stat(h.settingsPath); !os.IsNotExist(err) {
		t.Error("a cancelled menu wrote settings")
	}
	// Ctrl+D closes a menu like Esc; Ctrl+C in a menu exits.
	h.key(tcell.KeyF6)
	h.waitText("ANSWER STYLE ]")
	h.key(tcell.KeyCtrlD)
	h.waitFor("closed", func(s string) bool { return !strings.Contains(s, "ANSWER STYLE ]") })
	h.key(tcell.KeyF6)
	h.waitText("ANSWER STYLE ]")
	h.key(tcell.KeyCtrlC)
	if code := h.exitCode(); code != ExitInterrupted {
		t.Errorf("code %d", code)
	}
	if bottom, _ := h.screen.counts(); bottom != 0 {
		t.Error("the bottom-right cell was written")
	}
}

func ollamaModels(ids ...string) []llm.Model {
	models := make([]llm.Model, len(ids))
	for i, id := range ids {
		models[i] = llm.Model{ID: id, Details: "4B Q4_K_M"}
	}
	return models
}

func TestModelMenuScrolls(t *testing.T) {
	var ids []string
	for i := range 30 {
		ids = append(ids, "model-"+string(rune('a'+i%26))+string(rune('0'+i/26)))
	}
	fake := &fakeProvider{models: ollamaModels(ids...)}
	h := start(t, fake, setup{model: "model-c0", cols: 80, rows: 16}) // 9 rows: 7 items visible
	h.key(tcell.KeyF2)
	h.waitText("SELECT OLLAMA MODEL")
	h.waitText("3/30 Enter=OK")
	if !strings.Contains(h.text(), "* model-c0") || !strings.Contains(h.text(), "4B Q4_K_M") {
		t.Errorf("screen:\n%s", h.text())
	}
	h.key(tcell.KeyEnd)
	h.waitText("30/30 Enter=OK")
	if strings.Contains(h.text(), "* model-c0") || !strings.Contains(h.text(), "model-d1") {
		t.Errorf("not scrolled:\n%s", h.text())
	}
	h.key(tcell.KeyPgUp) // 6 up
	h.waitText("24/30 Enter=OK")
	h.key(tcell.KeyEsc)
	h.waitFor("closed", func(s string) bool { return !strings.Contains(s, "SELECT OLLAMA") })
}

func TestProviderMenu(t *testing.T) {
	openai := &fakeProvider{name: "openai", defaultModel: "gpt-6-luna"}
	fake := &fakeProvider{defaultModel: "qwen3.5:4b", turns: []turnScript{events(completed("Ollama answer", 3, 2)), events(completed("Back", 1, 1))}}
	h := start(t, fake, setup{streamed: true, others: map[string]*fakeProvider{"openai": openai}})
	h.submit("Q1")
	h.waitText("AI > Ollama answer")

	h.key(tcell.KeyF3)
	h.waitText("SELECT PROVIDER")
	if !strings.Contains(h.text(), "|   OpenAI") || !strings.Contains(h.text(), "| * Ollama") {
		t.Errorf("menu:\n%s", h.text())
	}
	h.key(tcell.KeyEnter) // the current provider: nothing happens
	h.waitFor("closed", func(s string) bool { return !strings.Contains(s, "SELECT PROVIDER") })
	if len(openai.calls)+len(fake.calls) != 0 {
		t.Errorf("requests: %v %v", openai.calls, fake.calls)
	}

	// Unavailable: everything stays.
	for _, reason := range []error{
		errors.New("OPENAI_API_KEY is not set. Export it in your environment and try again."),
		errors.New("OpenAI request failed with HTTP 500."),
	} {
		openai.unavailable = reason
		h.key(tcell.KeyF3)
		h.waitText("SELECT PROVIDER")
		h.key(tcell.KeyUp)
		h.key(tcell.KeyEnter)
		h.waitText("OpenAI is not available: " + reason.Error())
	}
	if h.sess.Selection() != (storage.Selection{Provider: "ollama", Model: "qwen3.5:4b"}) || !strings.Contains(h.text(), "AI > Ollama answer") {
		t.Errorf("switched: %v", h.sess.Selection())
	}

	openai.unavailable = nil
	h.key(tcell.KeyF3)
	h.waitText("SELECT PROVIDER")
	h.key(tcell.KeyUp)
	h.key(tcell.KeyEnter)
	h.waitText("Now using OpenAI / gpt-6-luna. Started a new conversation; the previous one stays saved.")
	if strings.Contains(h.text(), "Ollama answer") || !strings.Contains(h.text(), " OpenAI ") || !strings.Contains(h.text(), " gpt-6-luna") {
		t.Errorf("screen:\n%s", h.text())
	}
	if h.engine.Provider != llm.Provider(openai) || h.engine.Model != "gpt-6-luna" {
		t.Error("the engine was not switched")
	}

	// Back to Ollama (an injected provider of that name is the original).
	h.key(tcell.KeyF3)
	h.waitText("SELECT PROVIDER")
	h.key(tcell.KeyDown)
	h.key(tcell.KeyEnter)
	h.waitText("Now using Ollama / qwen3.5:4b. Restored its conversation (2 messages).")
	h.submit("Q2")
	h.waitText("AI > Back")
	if sent, _ := fake.state(); len(sent[1]) != 3 || sent[1][0].Content != "Q1" {
		t.Errorf("context %v", sent[1])
	}
}

func TestProviderCheckCancelled(t *testing.T) {
	openai := &fakeProvider{name: "openai", defaultModel: "gpt-6-luna", gate: make(chan struct{})}
	h := start(t, &fakeProvider{}, setup{others: map[string]*fakeProvider{"openai": openai}})
	h.key(tcell.KeyF3)
	h.waitText("SELECT PROVIDER")
	h.key(tcell.KeyUp)
	h.key(tcell.KeyEnter)
	h.waitStatus("CONNECTING | Checking OpenAI")
	h.typeText("typing works")
	h.waitText("YOU > typing works")
	h.key(tcell.KeyEnter) // not sent while checking
	h.waitStatus("WARNING | Wait for the current check or press Ctrl+C")
	h.key(tcell.KeyF6)
	h.key(tcell.KeyCtrlD) // never exits during a check
	h.key(tcell.KeyCtrlC)
	h.waitText("| Cancelled.")
	h.waitStatus("CANCELLED")
	close(openai.gate) // a late answer must not switch
	time.Sleep(50 * time.Millisecond)
	h.key(tcell.KeyLeft)
	h.waitStatus("READY")
	if h.sess.Selection().Provider != "ollama" || strings.Contains(h.text(), "Now using") {
		t.Errorf("switched after cancel:\n%s", h.text())
	}
}

func TestModelMenuOutcomes(t *testing.T) {
	gate := make(chan struct{})
	fake := &fakeProvider{models: ollamaModels("qwen3.5:4b", "llama3.2:3b"), gate: gate}
	h := start(t, fake, setup{})
	h.key(tcell.KeyF2)
	h.waitStatus("CONNECTING | Loading Ollama models")
	h.key(tcell.KeyCtrlC) // cancel the discovery
	h.waitStatus("CANCELLED")
	close(gate) // the late result must not open the menu
	time.Sleep(50 * time.Millisecond)
	if strings.Contains(h.text(), "SELECT OLLAMA MODEL") {
		t.Error("a cancelled discovery opened the menu")
	}

	h.key(tcell.KeyF2)
	h.waitText("SELECT OLLAMA MODEL")
	h.key(tcell.KeyEnter) // the current model: nothing happens
	h.waitFor("closed", func(s string) bool { return !strings.Contains(s, "SELECT OLLAMA") })
	if strings.Contains(h.text(), "Now using") {
		t.Error("same model switched")
	}

	fake.models = nil
	h.key(tcell.KeyF2)
	h.waitText("No selectable Ollama models were found.")
	fake.listErr = &llm.Error{Kind: llm.KindConnection, Message: "Ollama is not reachable."}
	h.key(tcell.KeyF2)
	h.waitText("Error: Ollama is not reachable.")
	h.waitStatus("ERROR | Ollama is not reachable.")
	if h.sess.Selection().Model != "qwen3.5:4b" {
		t.Error("selection changed")
	}
}

// Switching A → B → A restores each conversation, sends only its own
// messages, and leaves restored files byte for byte unchanged.
func TestSwitchingModels(t *testing.T) {
	var fileA string
	prepare := func(dir string) {
		created := time.Date(2026, 10, 20, 9, 30, 0, 0, time.UTC)
		c := storage.Conversation{ID: "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f", CreatedAt: created, UpdatedAt: created,
			Provider: "ollama", Model: "llama3.2:3b",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "B question"}, {Role: llm.RoleAssistant, Content: "B answer"}},
			Requests: []storage.UsageRecord{{At: created, Provider: "ollama", Model: "llama3.2:3b", Status: storage.StatusCompleted,
				Usage: llm.Usage{InputTokens: intp(40), OutputTokens: intp(2), TotalTokens: intp(42)}}}}
		if err := storage.NewHistoryStore(dir).Save(c); err != nil {
			t.Fatal(err)
		}
		fileA = filepath.Join(dir, c.ID+".json")
	}
	fake := &fakeProvider{models: ollamaModels("qwen3.5:4b", "llama3.2:3b"),
		turns: []turnScript{events(completed("A answer", 7, 1)), events(completed("B2", 50, 1)), events(completed("A2", 9, 1))}}
	h := start(t, fake, setup{streamed: true, prepare: prepare})
	before, _ := os.ReadFile(fileA)
	h.submit("A question")
	h.waitText("AI > A answer")

	h.key(tcell.KeyF2)
	h.waitText("SELECT OLLAMA MODEL")
	h.key(tcell.KeyDown)
	h.key(tcell.KeyEnter)
	h.waitText("Now using Ollama / llama3.2:3b. Restored its conversation (2 messages).")
	if strings.Contains(h.text(), "A answer") || !strings.Contains(h.text(), "AI > B answer") || !strings.Contains(h.text(), " in 40 out 2") {
		t.Errorf("screen:\n%s", h.text())
	}
	if after, _ := os.ReadFile(fileA); !reflect.DeepEqual(before, after) {
		t.Error("selecting a model rewrote its conversation")
	}
	h.submit("B again")
	h.waitText("AI > B2")

	h.key(tcell.KeyF2)
	h.waitText("SELECT OLLAMA MODEL")
	h.key(tcell.KeyHome)
	h.key(tcell.KeyEnter)
	h.waitText("Now using Ollama / qwen3.5:4b. Restored its conversation (2 messages).")
	h.submit("A again")
	h.waitText("AI > A2")

	sent, _ := fake.state()
	if got := sent[1]; len(got) != 3 || got[0].Content != "B question" {
		t.Errorf("B context %v", got)
	}
	if got := sent[2]; len(got) != 3 || got[0].Content != "A question" {
		t.Errorf("A context %v", got)
	}
	if reqs := fake.requests; reqs[1].Model != "llama3.2:3b" || reqs[2].Model != "qwen3.5:4b" {
		t.Errorf("models %q %q", reqs[1].Model, reqs[2].Model)
	}
	convs := h.saved()
	if len(convs) != 2 {
		t.Fatalf("saved %d", len(convs))
	}
	for _, c := range convs {
		if len(c.Messages) != 4 || len(c.Requests) != 2 {
			t.Errorf("%s: %d messages %d records", c.Model, len(c.Messages), len(c.Requests))
		}
	}
	active, _ := os.ReadFile(filepath.Join(filepath.Dir(h.historyDir), "active.json"))
	if !strings.Contains(string(active), `"qwen3.5:4b"`) {
		t.Errorf("active.json %s", active)
	}
}

func TestSwitchingWithoutHistory(t *testing.T) {
	openai := &fakeProvider{name: "openai", defaultModel: "gpt-6-luna", turns: []turnScript{events(completed("O", 1, 1))}}
	fake := &fakeProvider{models: ollamaModels("qwen3.5:4b", "llama3.2:3b"), turns: []turnScript{events(completed("A", 1, 1))}}
	h := start(t, fake, setup{streamed: true, noHistory: true, others: map[string]*fakeProvider{"openai": openai}})
	h.submit("Q")
	h.waitText("AI > A")
	h.key(tcell.KeyF2)
	h.waitText("SELECT OLLAMA MODEL")
	h.key(tcell.KeyDown)
	h.key(tcell.KeyEnter)
	h.waitText("Now using Ollama / llama3.2:3b. Started a new conversation; the previous messages are not sent to the new model.")
	h.key(tcell.KeyF3)
	h.waitText("SELECT PROVIDER")
	h.key(tcell.KeyHome)
	h.key(tcell.KeyEnter)
	h.waitText("Now using OpenAI / gpt-6-luna. Started a new conversation.")
	h.submit("Q2")
	h.waitText("AI > O")
	if sent, _ := openai.state(); len(sent[0]) != 1 {
		t.Errorf("context %v", sent[0])
	}
	h.key(tcell.KeyF6)
	h.waitText("ANSWER STYLE ]")
	h.key(tcell.KeyHome)
	h.key(tcell.KeyEnter)
	h.waitText("Answer style: Brief. Applies from the next message.")
	if _, err := os.Stat(filepath.Dir(h.historyDir)); !os.IsNotExist(err) {
		t.Errorf("--no-history touched the state directory: %v", err)
	}
	if data, _ := os.ReadFile(h.settingsPath); !strings.Contains(string(data), `"answer_style": "brief"`) {
		t.Errorf("settings not saved with --no-history: %s", data)
	}
}

// F2–F6 never open while a response is generated or a cancelled one is
// still closing; the typed text stays.
func TestSettingsBlockedWhileBusy(t *testing.T) {
	stuck := make(chan struct{})
	gate := make(chan struct{})
	fake := &fakeProvider{turns: []turnScript{
		gated(gate, deltas("Working")),
		func(ctx context.Context, yield func(llm.StreamEvent) bool) {
			yield(llm.TextDelta{Text: "Part"})
			<-stuck // ignores the cancellation
		},
	}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1")
	h.waitText("AI > Working")
	h.typeText("kept")
	for _, k := range []tcell.Key{tcell.KeyF2, tcell.KeyF3, tcell.KeyF4, tcell.KeyF5, tcell.KeyF6} {
		h.key(k)
		h.waitStatus("WARNING | Settings cannot change while a response is generated")
	}
	if strings.Contains(h.text(), "SELECT") || !strings.Contains(h.text(), "YOU > kept") {
		t.Errorf("screen:\n%s", h.text())
	}
	h.key(tcell.KeyCtrlC)
	h.waitStatus("CANCELLED")
	close(gate)
	h.key(tcell.KeyCtrlU)

	h.submit("Q2")
	h.waitText("AI > Part")
	h.key(tcell.KeyCtrlC)
	h.waitStatus("CANCELLED")
	h.key(tcell.KeyF6) // the worker is still closing
	h.waitStatus("WARNING | The cancelled response is still closing; try again shortly")
	if strings.Contains(h.text(), "ANSWER STYLE ]") {
		t.Error("menu opened while closing")
	}
	close(stuck)
	h.key(tcell.KeyF6)
	h.waitText("ANSWER STYLE ]")
}
