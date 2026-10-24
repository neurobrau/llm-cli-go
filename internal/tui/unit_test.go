package tui

import (
	"reflect"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

func TestLayout(t *testing.T) {
	l, ok := NewLayout(80, 25)
	if !ok {
		t.Fatal("80x25 too small")
	}
	x, panel := l.PanelX()
	got := []int{x, l.ConversationWidth(), l.ConversationHeight(), l.StatusSeparatorRow(), l.StatusRow(),
		l.InputSeparatorRow(), l.InputRow(), l.BottomRow(), l.HelpRow(), l.InnerWidth()}
	if want := []int{58, 55, 18, 19, 20, 21, 22, 23, 24, 76}; !panel || !reflect.DeepEqual(got, want) {
		t.Errorf("80x25: %v, want %v", got, want)
	}
	if l, _ := NewLayout(63, 20); func() bool { _, ok := l.PanelX(); return ok }() || l.ConversationWidth() != 59 {
		t.Error("63 columns: the panel must be hidden and the conversation use C-4")
	}
	if l, _ := NewLayout(64, 20); func() bool { x, ok := l.PanelX(); return !ok || x != 42 }() {
		t.Error("64 columns: panel at 42")
	}
	if _, ok := NewLayout(40, 12); !ok {
		t.Error("40x12 is the minimum")
	}
	for _, size := range [][2]int{{39, 12}, {40, 11}, {0, 0}} {
		if _, ok := NewLayout(size[0], size[1]); ok {
			t.Errorf("%v accepted", size)
		}
	}
}

func TestHorizontalRuleAndFit(t *testing.T) {
	if got := HorizontalRule(20, "Status"); got != "+-[ Status ]-------+" {
		t.Errorf("%q", got)
	}
	if got := HorizontalRule(8, "A long title"); got != "+-[ A l+" {
		t.Errorf("%q", got)
	}
	if got := HorizontalRule(5, ""); got != "+---+" {
		t.Errorf("%q", got)
	}
	if got := fit(" gpt-6-luna-2026-10-22", 18); got != " gpt-6-luna-2026-~" {
		t.Errorf("%q", got)
	}
}

func TestInputEditing(t *testing.T) {
	var in inputLine
	for _, r := range "Grüß Gott" {
		in.insert(r)
	}
	in.home()
	in.delete() // G
	in.end()
	in.backspace() // t
	in.left()
	in.left()
	in.insert('ö')
	if in.String() != "rüß Göot" || in.cursor != 6 {
		t.Errorf("%q cursor %d", in.String(), in.cursor)
	}
	in.home()
	in.backspace() // nothing before the cursor
	in.left()
	in.end()
	in.right()
	in.delete() // nothing after it
	if in.String() != "rüß Göot" || in.cursor != 8 {
		t.Errorf("%q cursor %d", in.String(), in.cursor)
	}
	in.clear()
	if in.String() != "" || in.cursor != 0 {
		t.Error("clear")
	}
}

func TestInputHorizontalScroll(t *testing.T) {
	var in inputLine
	in.set("Привет, мир") // 11 runes, cursor after the end
	if text, x := in.view(5); text != " мир" || x != 4 {
		t.Errorf("end: %q %d", text, x)
	}
	in.home()
	if text, x := in.view(5); text != "Приве" || x != 0 {
		t.Errorf("home: %q %d", text, x)
	}
	for range 7 {
		in.right()
	}
	if text, x := in.view(5); text != "вет, " || x != 4 {
		t.Errorf("middle: %q %d", text, x)
	}
	if text, x := in.view(40); text != "Привет, мир" || x != 7 {
		t.Errorf("wide: %q %d", text, x)
	}
}

func count(n int) *int { return &n }

func TestSettingsPanel(t *testing.T) {
	sel := storage.Selection{Provider: "ollama", Model: "qwen3.5:4b"}
	rows := settingsPanel(sel, "off", chat.Settings{AnswerStyle: "normal"}, nil)
	var got []string
	for _, r := range rows {
		got = append(got, r.text)
	}
	want := []string{"F2 MODEL", " qwen3.5:4b", "F3 PROVIDER", " Ollama", "F4 REASONING", " off", "F5 MAX TOKENS",
		" default", "F6 ANSWER STYLE", " normal", "TOKENS", " -", "COST (EST.)", " -"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("empty:\n%q\nwant\n%q", got, want)
	}

	records := []storage.UsageRecord{
		{Provider: "openai", Model: "gpt-6-luna-2026-10-22", Status: storage.StatusCompleted,
			Usage: llm.Usage{InputTokens: count(12_345_678), OutputTokens: count(30),
				TotalTokens: count(12_345_708), CachedInputTokens: count(600), ReasoningTokens: count(50_000)}},
		{Provider: "openai", Model: "gpt-6-luna-2026-10-22", Status: storage.StatusIncomplete,
			Usage: llm.Usage{InputTokens: count(5), OutputTokens: count(7)}},
	}
	rows = settingsPanel(storage.Selection{Provider: "openai", Model: "gpt-6-luna-2026-10-22"},
		"", chat.Settings{MaxOutputTokens: 512, AnswerStyle: "brief"}, records)
	got = nil
	for _, r := range rows {
		got = append(got, r.text)
	}
	want = []string{"F2 MODEL", " gpt-6-luna-2026-~", "F3 PROVIDER", " OpenAI", "F4 REASONING", " model default",
		"F5 MAX TOKENS", " 512", "F6 ANSWER STYLE", " brief", "TOKENS", " in 12.3M out 37", " total 12345708", // exact when it fits
		" cached 600", " reasoning 50k", " context 12.3M", "COST (EST.)", " ~$2.4691"} // tier and cache writes not reported
	if !reflect.DeepEqual(got, want) {
		t.Errorf("usage:\n%q\nwant\n%q", got, want)
	}
	if rows[0].role != roleBorder || rows[1].role != roleText {
		t.Error("roles")
	}
	// A count never reported is "?".
	rows = settingsPanel(sel, "", chat.Settings{}, []storage.UsageRecord{{Status: storage.StatusCompleted, Usage: llm.Usage{OutputTokens: count(3)}}})
	if rows[11].text != " in ? out 3" || rows[12].text != " total ?" {
		t.Errorf("unknown counts %q %q", rows[11].text, rows[12].text)
	}
}

func TestCompact(t *testing.T) {
	for n, want := range map[int]string{9999: "9999", 10_000: "10k", 12_345: "12.3k", 999_999: "1M",
		4_560_000: "4.56M", 2_000_000_000: "2G", 123_456_789: "123M"} {
		if got := compact(&n); got != want {
			t.Errorf("compact(%d) = %q, want %q", n, got, want)
		}
	}
	if compact(nil) != "?" {
		t.Error("nil")
	}
}

func TestDoneStatus(t *testing.T) {
	ollama := llm.Response{Usage: &llm.Usage{OutputTokens: count(30)},
		Timing: llm.Timing{Total: 7170 * time.Millisecond, Load: 200 * time.Millisecond, Eval: 1880 * time.Millisecond}}
	if got := doneStatus(ollama, 9*time.Second, time.Second).String(); got != "DONE | 7.17s total | load 0.20s | generation 1.88s | 15.96 tok/s" {
		t.Errorf("ollama %q", got)
	}
	if got := doneStatus(llm.Response{}, 1234*time.Millisecond, 456*time.Millisecond).String(); got != "DONE | 1.23s observed | first text after 0.46s" {
		t.Errorf("observed %q", got)
	}
	if got := doneStatus(llm.Response{}, 1234*time.Millisecond, 0).String(); got != "DONE | 1.23s observed" {
		t.Errorf("not streamed %q", got)
	}
	if got := activeStatus(stateGenerating, 1340*time.Millisecond, "Thinking").String(); got != "GENERATING | Thinking | 1.3s elapsed | Ctrl+C Cancel" {
		t.Errorf("active %q", got)
	}
}
