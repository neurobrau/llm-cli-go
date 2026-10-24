package tui

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/gdamore/tcell/v2"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// settingsFile returns settings.json as generic JSON (nil if missing).
func (h *harness) settingsFile() map[string]any {
	h.t.Helper()
	data, err := os.ReadFile(h.settingsPath)
	if os.IsNotExist(err) {
		return nil
	}
	var v map[string]any
	if err := json.Unmarshal(data, &v); err != nil {
		h.t.Fatalf("settings.json: %v\n%s", err, data)
	}
	return v
}

// modelEntry is the saved entry of one model in settings.json.
func (h *harness) modelEntry(provider, model string) map[string]any {
	h.t.Helper()
	v := h.settingsFile()
	providers, _ := v["providers"].(map[string]any)
	p, _ := providers[provider].(map[string]any)
	models, _ := p["models"].(map[string]any)
	entry, _ := models[model].(map[string]any)
	return entry
}

// choose opens a menu with key, moves with keys and presses Enter.
func (h *harness) choose(key tcell.Key, title string, moves ...tcell.Key) {
	h.t.Helper()
	h.key(key)
	h.waitText(title)
	for _, k := range moves {
		h.key(k)
	}
	h.key(tcell.KeyEnter)
	h.waitFor("menu closed", func(s string) bool { return !strings.Contains(s, title) })
}

var lunaSupport = llm.ReasoningSupport{Values: []string{"none", "low", "medium", "high", "xhigh", "max"}, Known: true,
	Default: "none", ModelDefault: "medium"}

func TestReasoningOllamaOnOff(t *testing.T) {
	fake := &fakeProvider{turns: []turnScript{events(completed("A", 1, 1)), events(completed("B", 1, 1))}}
	h := start(t, fake, setup{streamed: true})
	h.key(tcell.KeyF4)
	h.waitText("REASONING ]")
	if !strings.Contains(h.text(), "| * Off") || !strings.Contains(h.text(), "|   On ") || strings.Contains(h.text(), "Model default") {
		t.Errorf("menu:\n%s", h.text())
	}
	h.key(tcell.KeyDown)
	h.key(tcell.KeyEnter)
	h.waitText("Reasoning for qwen3.5:4b: On. Applies from the next message.")
	if got := h.modelEntry("ollama", "qwen3.5:4b"); !reflect.DeepEqual(got, map[string]any{"reasoning": "on"}) {
		t.Errorf("saved %v", got)
	}
	if !strings.Contains(h.text(), "|  on ") {
		t.Errorf("panel:\n%s", h.text())
	}
	h.submit("Q1")
	h.waitText("AI > A")
	if r := fake.lastRequest().Reasoning; r != (llm.Reasoning{Effort: "on", Explicit: true}) {
		t.Errorf("request reasoning %+v", r)
	}

	// Off is saved, so it sticks across launches.
	h.choose(tcell.KeyF4, "REASONING ]", tcell.KeyHome)
	h.waitText("Reasoning for qwen3.5:4b: Off.")
	if got := h.modelEntry("ollama", "qwen3.5:4b"); got["reasoning"] != "off" {
		t.Errorf("saved %v", got)
	}
	// Choosing the current, saved value changes nothing.
	os.Remove(h.settingsPath)
	h.choose(tcell.KeyF4, "REASONING ]")
	h.submit("Q2")
	h.waitText("AI > B")
	if _, err := os.Stat(h.settingsPath); !os.IsNotExist(err) {
		t.Error("an unchanged choice was saved")
	}
	if strings.Count(h.conversation(), "Reasoning for") != 2 {
		t.Errorf("entries:\n%s", h.conversation())
	}
	if r := fake.lastRequest().Reasoning; r != (llm.Reasoning{Effort: "off", Explicit: true}) {
		t.Errorf("request reasoning %+v", r)
	}
}

func TestReasoningKinds(t *testing.T) {
	t.Run("known OpenAI model", func(t *testing.T) {
		fake := &fakeProvider{name: "openai", support: map[string]llm.ReasoningSupport{"gpt-6-luna": lunaSupport},
			turns: []turnScript{events(completed("A", 1, 1))}}
		h := start(t, fake, setup{model: "gpt-6-luna", reasoning: "none"})
		h.key(tcell.KeyF4)
		h.waitText("REASONING ]")
		for _, want := range []string{"|   Model default  ", "no effort sent: medium |", "| * None", "|   Low", "|   Medium",
			"|   High", "|   XHigh", "|   Max", "2/7 Enter=OK"} {
			if !strings.Contains(h.text(), want) {
				t.Errorf("menu lacks %q:\n%s", want, h.text())
			}
		}
		h.key(tcell.KeyHome)
		h.key(tcell.KeyEnter)
		h.waitText("Reasoning for gpt-6-luna: Model default (no effort parameter is sent). Applies from the next message.")
		if got := h.modelEntry("openai", "gpt-6-luna"); got["reasoning"] != "default" {
			t.Errorf("saved %v", got)
		}
		if !strings.Contains(h.text(), "|  model default") {
			t.Errorf("panel:\n%s", h.text())
		}
		h.submit("Q")
		h.waitText("AI > A")
		if r := fake.lastRequest().Reasoning; r != (llm.Reasoning{Explicit: true}) {
			t.Errorf("request reasoning %+v", r)
		}
	})
	t.Run("unknown support", func(t *testing.T) {
		fake := &fakeProvider{name: "openai", support: map[string]llm.ReasoningSupport{"my-model": {}}}
		h := start(t, fake, setup{model: "my-model", reasoning: "-"})
		h.key(tcell.KeyF4)
		h.waitText("Reasoning support for my-model is not confirmed, so only the model's default is offered. --reasoning-effort sends a value explicitly.")
		h.waitText("| * Model default (support unknown)")
		h.key(tcell.KeyEsc)
	})
	t.Run("no reasoning", func(t *testing.T) {
		fake := &fakeProvider{support: map[string]llm.ReasoningSupport{"qwen3.5:4b": {Known: true}}}
		h := start(t, fake, setup{})
		h.key(tcell.KeyF4)
		h.waitText("qwen3.5:4b does not support reasoning.")
		if strings.Contains(h.text(), "REASONING ]") {
			t.Error("menu opened")
		}
	})
	t.Run("named levels", func(t *testing.T) {
		fake := &fakeProvider{support: map[string]llm.ReasoningSupport{"qwen3.5:4b": {Known: true, Values: []string{"low", "medium", "high"}}}}
		h := start(t, fake, setup{reasoning: "-"})
		h.key(tcell.KeyF4)
		h.waitText("REASONING ]")
		if !strings.Contains(h.text(), "Model default  no effort sent |") || !strings.Contains(h.text(), "|   Medium") {
			t.Errorf("menu:\n%s", h.text())
		}
		h.key(tcell.KeyEsc)
	})
	t.Run("failed check", func(t *testing.T) {
		fake := &fakeProvider{supportErr: &llm.Error{Kind: llm.KindConnection, Message: "Ollama is not reachable."}}
		h := start(t, fake, setup{})
		h.key(tcell.KeyF4)
		h.waitText("Error: Ollama is not reachable.")
		if strings.Contains(h.text(), "REASONING ]") || h.settingsFile() != nil || h.engine.Settings.Reasoning != (llm.Reasoning{}) {
			t.Error("settings changed after a failed check")
		}
	})
}

func TestMaxTokensMenu(t *testing.T) {
	fake := &fakeProvider{limits: map[string]int{"qwen3.5:4b": 2048},
		turns: []turnScript{events(completed("A", 1, 1)), events(completed("B", 1, 1))}}
	h := start(t, fake, setup{streamed: true})
	h.typeText("my draft")
	h.key(tcell.KeyF5)
	h.waitText("MAX OUTPUT TOKENS ]")
	for _, want := range []string{"| * Default", "model default |", "|   256", "|   2048", "|   Custom...", "up to 2048 |", "1/6 Enter=OK"} {
		if !strings.Contains(h.text(), want) {
			t.Errorf("menu lacks %q:\n%s", want, h.text())
		}
	}
	if strings.Contains(h.text(), "4096") {
		t.Error("a preset above the limit is offered")
	}
	h.key(tcell.KeyEnter) // Default again: nothing to save
	h.waitFor("closed", func(s string) bool { return !strings.Contains(s, "MAX OUTPUT TOKENS ]") })
	if h.settingsFile() != nil {
		t.Error("an unchanged choice was saved")
	}

	h.choose(tcell.KeyF5, "MAX OUTPUT TOKENS ]", tcell.KeyDown, tcell.KeyDown) // 512
	h.waitText("Max output tokens for qwen3.5:4b: 512. Applies from the next message.")
	if got := h.modelEntry("ollama", "qwen3.5:4b"); !reflect.DeepEqual(got, map[string]any{"max_output_tokens": float64(512)}) {
		t.Errorf("saved %v", got)
	}
	h.choose(tcell.KeyF5, "MAX OUTPUT TOKENS ]", tcell.KeyHome) // Default: JSON null
	h.waitText("Max output tokens for qwen3.5:4b: model default.")
	if got, ok := h.modelEntry("ollama", "qwen3.5:4b")["max_output_tokens"]; !ok || got != nil {
		t.Errorf("Default saved as %v (present %v), want null", got, ok)
	}
	if !strings.Contains(h.text(), "YOU > my draft") {
		t.Error("the draft was lost")
	}

	// Custom: the draft is set aside; errors keep the prompt open.
	h.choose(tcell.KeyF5, "MAX OUTPUT TOKENS ]", tcell.KeyEnd)
	h.waitText("Type the max output tokens (1-2048) and press Enter; Esc cancels.")
	h.waitStatus("READY | Type a token limit and press Enter | Esc cancels")
	if !strings.Contains(h.text(), "| TOKENS > ") || strings.Contains(h.text(), "my draft") {
		t.Errorf("prompt:\n%s", h.text())
	}
	h.key(tcell.KeyF6) // ignored in the prompt
	for input, want := range map[string]string{
		"abc":                     "'abc' is not a positive whole number. Try again or press Esc.",
		"-5":                      "'-5' is not a positive whole number. Try again or press Esc.",
		"0":                       "Max output tokens must be a positive whole number, not 0. Try again or press Esc.",
		"4097":                    "4097 exceeds the limit of 2048 output tokens for qwen3.5:4b. Try again or press Esc.",
		"99999999999999999999999": "99999999999999999999999 exceeds the limit of 2048 output tokens",
	} {
		h.submit(input)
		h.waitText(want)
	}
	if strings.Contains(h.text(), "ANSWER STYLE ]") || !strings.Contains(h.text(), "| TOKENS > ") {
		t.Error("the prompt closed")
	}
	h.typeText("12")
	h.key(tcell.KeyEsc)
	h.waitText("Max output tokens unchanged.")
	h.waitText("| YOU > my draft")
	if h.engine.Settings.MaxOutputTokens != 0 {
		t.Error("Esc changed the setting")
	}

	h.choose(tcell.KeyF5, "MAX OUTPUT TOKENS ]", tcell.KeyEnd)
	h.submit("1500")
	h.waitText("Max output tokens for qwen3.5:4b: 1500. Applies from the next message.")
	h.waitText("| YOU > my draft")
	if got := h.modelEntry("ollama", "qwen3.5:4b"); got["max_output_tokens"] != float64(1500) {
		t.Errorf("saved %v", got)
	}
	h.key(tcell.KeyF5)
	h.waitText("| * Custom (1500)")
	h.key(tcell.KeyEsc)
	h.key(tcell.KeyCtrlU)
	h.submit("Q")
	h.waitText("AI > A")
	if n := fake.lastRequest().MaxOutputTokens; n != 1500 {
		t.Errorf("request max tokens %d", n)
	}
}

func TestMaxTokensUnknownLimit(t *testing.T) {
	h := start(t, &fakeProvider{}, setup{})
	h.key(tcell.KeyF5)
	h.waitText("any positive number |")
	if !strings.Contains(h.text(), "|   4096") {
		t.Errorf("menu:\n%s", h.text())
	}
	h.key(tcell.KeyEnd)
	h.key(tcell.KeyEnter)
	h.waitText("Type the max output tokens and press Enter; Esc cancels.")
	h.submit("99999999999999999999999")
	h.waitText("'99999999999999999999999' is too large. Try again or press Esc.")
	h.submit("100000")
	h.waitText("Max output tokens for qwen3.5:4b: 100000.")
}

func TestAnswerStyle(t *testing.T) {
	fake := &fakeProvider{turns: []turnScript{events(completed("A", 1, 1)), events(completed("B", 1, 1))}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1")
	h.waitText("AI > A")
	if fake.lastRequest().Instructions != "" {
		t.Error("normal style sent instructions")
	}
	h.choose(tcell.KeyF6, "ANSWER STYLE ]", tcell.KeyEnd)
	h.waitText("Answer style: Detailed. Applies from the next message.")
	if v := h.settingsFile(); v["answer_style"] != "detailed" || v["schema_version"] != float64(1) || len(v["providers"].(map[string]any)) != 0 {
		t.Errorf("settings %v", v)
	}
	if !strings.Contains(h.text(), "AI > A") || !strings.Contains(h.text(), "|  detailed") {
		t.Errorf("screen:\n%s", h.text())
	}
	h.submit("Q2")
	h.waitText("AI > B")
	req := fake.lastRequest()
	if req.Instructions != llm.StyleInstructions(llm.StyleDetailed) || len(req.Messages) != 3 {
		t.Errorf("request %+v", req)
	}
	for _, c := range h.saved() {
		for _, m := range c.Messages {
			if strings.Contains(m.Content, "comprehensive") {
				t.Error("the style instruction was stored in the conversation")
			}
		}
	}
	os.Remove(h.settingsPath)
	h.choose(tcell.KeyF6, "ANSWER STYLE ]") // unchanged
	if h.settingsFile() != nil {
		t.Error("an unchanged choice was saved")
	}
}

// A choice made in the UI replaces the command-line value for the rest of
// the run; other flags stay in force, also after a switch.
func TestChoicesReplaceCommandLineValues(t *testing.T) {
	settings := `{"schema_version": 1, "answer_style": "normal", "providers": {"ollama": {"models": {
	  "llama3.2:3b": {"max_output_tokens": 128},
	  "qwen3.5:4b": {"reasoning": "on"}}}}}`
	fake := &fakeProvider{models: ollamaModels("qwen3.5:4b", "llama3.2:3b"),
		support: map[string]llm.ReasoningSupport{"llama3.2:3b": {Known: true}},
		turns:   []turnScript{events(completed("A", 1, 1)), events(completed("B", 1, 1))}}
	overrides := chat.Overrides{MaxOutputTokens: 300, AnswerStyle: llm.StyleBrief}
	h := start(t, fake, setup{streamed: true, settingsJSON: settings, overrides: overrides, reasoning: "on",
		settings: chat.Settings{MaxOutputTokens: 300, AnswerStyle: llm.StyleBrief, Reasoning: llm.Reasoning{Effort: "on", Explicit: true}}})
	h.key(tcell.KeyF5)
	h.waitText("| * Custom (300)")
	h.key(tcell.KeyHome)
	h.key(tcell.KeyEnter)
	h.waitText("Max output tokens for qwen3.5:4b: model default.")

	h.choose(tcell.KeyF2, "SELECT OLLAMA MODEL", tcell.KeyDown)
	h.waitText("Now using Ollama / llama3.2:3b.")
	h.submit("Q1")
	h.waitText("AI > A")
	req := fake.lastRequest()
	// llama: its own saved 128 (the flag was replaced), the --answer-style flag still applies.
	if req.MaxOutputTokens != 128 || req.Instructions != llm.StyleInstructions(llm.StyleBrief) || req.Reasoning != (llm.Reasoning{}) {
		t.Errorf("llama request %+v", req)
	}
	h.choose(tcell.KeyF2, "SELECT OLLAMA MODEL", tcell.KeyHome)
	h.waitText("Now using Ollama / qwen3.5:4b.")
	h.submit("Q2")
	h.waitText("AI > B")
	req = fake.lastRequest()
	if req.MaxOutputTokens != 0 || req.Reasoning != (llm.Reasoning{Effort: "on", Explicit: true}) {
		t.Errorf("qwen request %+v", req)
	}
	entry := h.modelEntry("ollama", "qwen3.5:4b")
	if v, ok := entry["max_output_tokens"]; !ok || v != nil || entry["reasoning"] != "on" {
		t.Errorf("saved %v", entry)
	}
	if got := h.modelEntry("ollama", "llama3.2:3b"); got["max_output_tokens"] != float64(128) {
		t.Errorf("another model's entry changed: %v", got)
	}
	if v := h.settingsFile(); v["answer_style"] != "normal" {
		t.Errorf("the --answer-style flag was saved: %v", v)
	}
}

// Saved values the new model does not support fall back with a notice,
// and the file is not rewritten.
func TestUnsupportedSavedValuesOnSwitch(t *testing.T) {
	settings := `{"schema_version": 1, "answer_style": "normal", "providers": {"ollama": {"models": {
	  "llama3.2:3b": {"reasoning": "high", "max_output_tokens": 9000}}}}}`
	fake := &fakeProvider{models: ollamaModels("qwen3.5:4b", "llama3.2:3b"), limits: map[string]int{"llama3.2:3b": 4096}}
	h := start(t, fake, setup{settingsJSON: settings})
	before, _ := os.ReadFile(h.settingsPath)
	h.choose(tcell.KeyF2, "SELECT OLLAMA MODEL", tcell.KeyDown)
	h.waitText("Now using Ollama / llama3.2:3b. Started a new conversation.")
	h.waitText("Saved reasoning 'High' is not available for llama3.2:3b")
	h.waitText("Saved setting ignored: 9000 exceeds the limit of 4096 output tokens for llama3.2:3b. Using the default.")
	if after, _ := os.ReadFile(h.settingsPath); string(after) != string(before) {
		t.Error("settings.json was rewritten")
	}
	if !strings.Contains(h.text(), "|  off ") || h.engine.Settings.MaxOutputTokens != 0 {
		t.Errorf("screen:\n%s", h.text())
	}
}

func TestCorruptSettingsAreNeverOverwritten(t *testing.T) {
	const corrupt = `{"schema_version": 1, "answer_style": `
	fake := &fakeProvider{turns: []turnScript{events(completed("A", 1, 1))}}
	h := start(t, fake, setup{streamed: true, settingsJSON: corrupt})
	h.choose(tcell.KeyF6, "ANSWER STYLE ]", tcell.KeyHome)
	h.waitText("Answer style: Brief. Applies from the next message.")
	h.waitText("Warning: Settings not saved: /")
	h.waitStatus("WARNING | Settings not saved")
	if data, _ := os.ReadFile(h.settingsPath); string(data) != corrupt {
		t.Errorf("settings.json changed: %q", data)
	}
	h.submit("Q")
	h.waitText("AI > A")
	if fake.lastRequest().Instructions != llm.StyleInstructions(llm.StyleBrief) {
		t.Error("the choice is not in effect")
	}
}

func TestSettingsSaveFailure(t *testing.T) {
	const valid = `{"schema_version": 1, "answer_style": "normal", "providers": {}}`
	// Not in an app directory, which a save would tighten to 0700 first.
	h := start(t, &fakeProvider{}, setup{settingsJSON: valid, settingsDir: "shared"})
	dir := filepath.Dir(h.settingsPath)
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0o700) })
	if f, err := os.CreateTemp(dir, "probe"); err == nil { // e.g. running as root
		f.Close()
		os.Remove(f.Name())
		t.Skip("the directory is still writable")
	}
	h.choose(tcell.KeyF6, "ANSWER STYLE ]", tcell.KeyEnd)
	h.waitText("Answer style: Detailed.")
	h.waitText("Warning: Could not save settings at /")
	if data, _ := os.ReadFile(h.settingsPath); string(data) != valid {
		t.Errorf("settings.json changed: %q", data)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Errorf("leftover files: %v", entries)
	}
	if h.engine.Settings.AnswerStyle != llm.StyleDetailed {
		t.Error("the choice is not in effect")
	}
}
