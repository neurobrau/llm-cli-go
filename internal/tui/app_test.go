package tui

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// baseline is the exact render of docs/tui-spec.md, section 2 (rows 14–18
// are elided there).
var baseline = map[int]string{
	0:  "+-[ NeuroBräu CLI 0.2.0 ]---------------------------------+-[ LLM SETTINGS ]---+",
	1:  "| YOU > What is a neural network?                         | F2 MODEL           |",
	2:  "|                                                         |  qwen3.5:4b        |",
	3:  "| AI > A neural network is a model made of layers of      | F3 PROVIDER        |",
	4:  "|      connected units that learn weights from data.      |  Ollama            |",
	5:  "|                                                         | F4 REASONING       |",
	6:  "|                                                         |  off               |",
	7:  "|                                                         | F5 MAX TOKENS      |",
	8:  "|                                                         |  default           |",
	9:  "|                                                         | F6 ANSWER STYLE    |",
	10: "|                                                         |  normal            |",
	11: "|                                                         | TOKENS             |",
	12: "|                                                         |  in 17 out 30      |",
	13: "|                                                         |  total 47          |",
	19: "+-[ Status ]----------------------------------------------+--------------------+",
	20: "| DONE | 7.17s total | load 0.20s | generation 1.88s | 15.96 tok/s             |",
	21: "+-[ Input ]--------------------------------------------------------------------+",
	22: "| YOU >                                                                        |",
	23: "+------------------------------------------------------------------------------+",
	24: " F1 /help | /reset /exit | F2-F6 settings | PgUp/PgDn scroll | Ctrl+C quit",
}

func ollamaAnswer() llm.StreamCompleted {
	c := completed("A neural network is a model made of layers of connected units that learn weights from data.", 17, 30)
	c.Response.Timing = llm.Timing{Total: 7170 * time.Millisecond, Load: 200 * time.Millisecond, Eval: 1880 * time.Millisecond}
	return c
}

func ollamaAnswerWith(text string) llm.StreamCompleted {
	c := ollamaAnswer()
	c.Response.Text = text
	return c
}

func TestBaselineRender(t *testing.T) {
	answer := ollamaAnswer()
	fake := &fakeProvider{turns: []turnScript{events(append(deltas("A neural network is a model ", "made of layers"), answer)...)}}
	h := start(t, fake, setup{streamed: true, title: "NeuroBräu CLI 0.2.0"})
	h.submit("What is a neural network?")
	h.waitStatus("DONE")
	rows := h.rows()
	for y, want := range baseline {
		if got := strings.TrimRight(rows[y], " "); got != strings.TrimRight(want, " ") {
			t.Errorf("row %d:\n got %q\nwant %q", y, got, want)
		}
	}
	// The rest of the panel: context, then the cost of local inference.
	for y, want := range map[int]string{14: " context 17", 15: "COST (EST.)", 16: " $0 (local)"} {
		if got := strings.TrimRight(string([]rune(rows[y])[60:78]), " "); got != want {
			t.Errorf("panel row %d: %q, want %q", y, got, want)
		}
	}

	// Colors: everything on black; green bold borders, blue bold YOU,
	// green bold AI, light gray text, green DONE.
	for _, c := range []struct {
		x, y   int
		r      rune
		fg     tcell.Color
		bold   bool
		reason string
	}{
		{0, 0, '+', tcell.ColorGreen, true, "border"},
		{2, 1, 'Y', tcell.ColorNavy, true, "user label"},
		{8, 1, 'W', tcell.ColorSilver, false, "text"},
		{2, 3, 'A', tcell.ColorGreen, true, "AI label"},
		{60, 1, 'F', tcell.ColorGreen, true, "panel heading"},
		{61, 2, 'q', tcell.ColorSilver, false, "panel value"},
		{2, 20, 'D', tcell.ColorGreen, true, "DONE"},
		{1, 24, 'F', tcell.ColorGreen, true, "help line"},
		{40, 10, ' ', tcell.ColorSilver, false, "background"},
	} {
		r, style := h.cell(c.x, c.y)
		fg, bg, attr := style.Decompose()
		if r != c.r || fg != c.fg || bg != tcell.ColorBlack || (attr&tcell.AttrBold != 0) != c.bold {
			t.Errorf("%s at %d,%d: %q fg=%v bg=%v attr=%v", c.reason, c.x, c.y, r, fg, bg, attr)
		}
	}
	if x, y, visible := h.screen.GetCursor(); x != 8 || y != 22 || !visible {
		t.Errorf("cursor %d,%d visible=%v", x, y, visible)
	}
	if bottom, _ := h.screen.counts(); bottom != 0 {
		t.Errorf("the bottom-right cell was written %d times", bottom)
	}
	// The status returns to READY with the next key press.
	h.key(tcell.KeyRight)
	h.waitStatus("READY | Conversation saved")
}

func TestSizes(t *testing.T) {
	fake := &fakeProvider{}
	h := start(t, fake, setup{cols: 63, rows: 20})
	if strings.Contains(h.text(), "LLM SETTINGS") || h.at(62, 0) != '+' || strings.Count(h.row(5), "|") != 2 {
		t.Errorf("63 columns show the panel:\n%s", h.text())
	}
	h.resize(64, 20)
	h.waitText("LLM SETTINGS")
	if h.at(42, 5) != '|' || h.at(42, 14) != '+' {
		t.Errorf("64 columns: panel separator not at 42:\n%s", h.text())
	}

	h.typeText("draft Grüß")
	h.resize(40, 12)
	h.waitFor("40x12 layout", func(s string) bool { return strings.Contains(s, "YOU > draft Grüß") })
	if h.row(11) != " F1 /help | /reset /exit | F2-F6 settin" { // clipped to C-1
		t.Errorf("help row %q", h.row(11))
	}

	h.resize(39, 11)
	h.waitText("Terminal too small.")
	if rows := h.rows(); strings.TrimRight(rows[1], " ") != "Need 40x12, have 39x11." || strings.TrimRight(rows[2], " ") != "Resize or press Ctrl+C." {
		t.Errorf("too small:\n%s", h.text())
	}
	if _, _, visible := h.screen.GetCursor(); visible {
		t.Error("cursor visible on the too-small screen")
	}
	h.typeText("!") // input is still accepted
	h.resize(10, 2)
	h.waitFor("tiny screen", func(string) bool { return h.row(1) == "Need 40x1" })
	if r := h.rows(); r[0] != "Terminal  " || r[1] != "Need 40x1 " { // the last column stays empty
		t.Errorf("clipping: %q", r)
	}

	h.resize(80, 25)
	h.waitText("YOU > draft Grüß!")
	if bottom, _ := h.screen.counts(); bottom != 0 {
		t.Errorf("the bottom-right cell was written %d times", bottom)
	}
	h.key(tcell.KeyCtrlC)
	if code := h.exitCode(); code != ExitInterrupted {
		t.Errorf("code %d", code)
	}
}

func TestStreamingShowsTextWhileGenerated(t *testing.T) {
	gate := make(chan struct{})
	fake := &fakeProvider{turns: []turnScript{
		gated(gate, []llm.StreamEvent{llm.ReasoningDelta{Text: "SECRET THOUGHT"}}),
		gated(gate, deltas("First "), append(deltas("second."), completed("First second.", 5, 2))...),
		events(completed("Third.", 9, 1)),
	}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1")
	h.waitStatus("GENERATING | Thinking | ")
	if strings.Contains(h.text(), "SECRET") || strings.Contains(h.text(), "AI >") {
		t.Errorf("reasoning shown:\n%s", h.text())
	}
	h.key(tcell.KeyCtrlC) // cancel the thinking request
	h.waitStatus("CANCELLED")

	h.submit("Q2")
	h.waitText("AI > First") // before the answer is complete
	if !strings.HasPrefix(h.status(), "GENERATING | ") || !strings.HasSuffix(h.status(), "s elapsed | Ctrl+C Cancel") {
		t.Errorf("status %q", h.status())
	}
	h.typeText("typing works") // input and redraw keep working
	h.waitText("YOU > typing works")
	close(gate)
	h.waitText("AI > First second.")
	h.waitStatus("DONE | ")
	if !strings.Contains(h.status(), "s observed | first text after ") {
		t.Errorf("status %q", h.status())
	}

	h.key(tcell.KeyCtrlU)
	h.submit("Q3")
	h.waitText("AI > Third.")
	sent, _ := fake.state()
	if got := sent[2]; len(got) != 3 || got[0].Content != "Q2" || got[1].Content != "First second." {
		t.Errorf("third context %v", got) // the canceled Q1 is not part of it
	}
	convs := h.saved()
	if len(convs) != 1 || len(convs[0].Messages) != 4 || len(convs[0].Requests) != 2 ||
		convs[0].Requests[0].Status != storage.StatusCompleted || convs[0].Provider != "ollama" {
		t.Errorf("saved %+v", convs)
	}
}

func TestCompleteAnswersWithoutStream(t *testing.T) {
	gate, gate2 := make(chan struct{}), make(chan struct{})
	fake := &fakeProvider{turns: []turnScript{
		gated(gate, nil, ollamaAnswerWith("The whole answer.")),
		events(llm.StreamIncomplete{Reason: "the max output token limit was reached", PartialText: "Cut",
			Usage: &llm.Usage{InputTokens: intp(1), OutputTokens: intp(2)}}),
		events(llm.StreamFailed{Err: &llm.Error{Kind: llm.KindConnection, Message: "Ollama is not reachable."}}),
		gated(gate2, nil),
	}}
	h := start(t, fake, setup{})
	h.submit("Q1")
	h.waitStatus("GENERATING | Waiting for the complete response | Ctrl+C Cancel")
	if strings.Contains(h.text(), "AI >") {
		t.Error("text shown before the complete answer")
	}
	close(gate)
	h.waitText("AI > The whole answer.")
	h.waitStatus("DONE | ")
	if s := h.status(); !strings.HasSuffix(s, "s observed") || strings.Contains(s, "total") { // Ollama's durations: streamed only
		t.Errorf("status %q", s)
	}

	h.submit("Q2")
	h.waitText("Incomplete: the max output token limit was reached. The partial answer above was not saved.")
	if !strings.Contains(h.text(), "AI > Cut") || h.status() != "WARNING | Response incomplete: the max output token limit was reached" {
		t.Errorf("incomplete:\n%s", h.text())
	}
	h.submit("Q3")
	h.waitText("Error: Ollama is not reachable.")
	if h.status() != "ERROR | Ollama is not reachable." {
		t.Errorf("status %q", h.status())
	}

	h.submit("Q4")
	h.waitStatus("GENERATING")
	h.key(tcell.KeyCtrlC)
	h.waitText("Request cancelled.")
	if h.status() != "CANCELLED" {
		t.Errorf("status %q", h.status())
	}
	h.waitFor("worker finished", func(string) bool { _, n := fake.state(); return n == 4 })

	convs := h.saved()
	if len(convs) != 1 || len(convs[0].Messages) != 2 || len(convs[0].Requests) != 2 ||
		convs[0].Requests[1].Status != storage.StatusIncomplete {
		t.Errorf("saved %+v", convs)
	}
}

func TestRestoredConversation(t *testing.T) {
	prepare := func(dir string) {
		store := storage.NewHistoryStore(dir)
		created := time.Date(2026, 10, 20, 9, 30, 0, 0, time.UTC)
		err := store.Save(storage.Conversation{ID: "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f",
			CreatedAt: created, UpdatedAt: created, Provider: "ollama", Model: "qwen3.5:4b",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Old question"}, {Role: llm.RoleAssistant, Content: "Old answer"}},
			Requests: []storage.UsageRecord{{At: created, Provider: "ollama", Model: "qwen3.5:4b", Status: storage.StatusCompleted,
				Usage: llm.Usage{InputTokens: intp(10), OutputTokens: intp(5), TotalTokens: intp(15)}}}})
		if err != nil {
			t.Fatal(err)
		}
	}
	fake := &fakeProvider{turns: []turnScript{events(completed("New answer", 20, 3))}}
	h := start(t, fake, setup{streamed: true, prepare: prepare, warnings: []string{"Skipped an invalid file."}})
	rows := h.rows()
	want := []string{"| YOU > Old question", "|", "| AI > Old answer", "|", "| Warning: Skipped an invalid file.", "|",
		"| Restored conversation from " + time.Date(2026, 10, 20, 9, 30, 0, 0, time.UTC).Local().Format("2006-01-02 15:04") + " (2"}
	for i, w := range want {
		if !strings.HasPrefix(rows[1+i], w) {
			t.Errorf("row %d %q, want %q", 1+i, rows[1+i], w)
		}
	}
	if !strings.Contains(h.text(), "messages). Type /reset to start a new") || !strings.Contains(rows[12], " in 10 out 5") {
		t.Errorf("screen:\n%s", h.text())
	}
	if _, style := h.cell(2, 5); style != newStyles(256)[roleError] {
		t.Error("warning not in the error style")
	}
	h.submit("New question")
	h.waitText("AI > New answer")
	sent, _ := fake.state()
	if len(sent[0]) != 3 || sent[0][0].Content != "Old question" {
		t.Errorf("context %v", sent[0])
	}
	h.waitFor("totals", func(s string) bool { return strings.Contains(s, " in 30 out 8") })
	if convs := h.saved(); len(convs) != 1 || len(convs[0].Messages) != 4 || convs[0].ID != "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f" {
		t.Errorf("saved %+v", convs)
	}
}

func TestCtrlCDuringStream(t *testing.T) {
	fake := &fakeProvider{turns: []turnScript{
		// Ignores the cancellation for a while and tries to send late
		// events, which must never be shown.
		func(ctx context.Context, yield func(llm.StreamEvent) bool) {
			if !yield(llm.TextDelta{Text: "Partial"}) {
				return
			}
			<-ctx.Done()
			for range 100 {
				if !yield(llm.TextDelta{Text: " LATE"}) {
					return
				}
			}
			yield(completed("Partial LATE", 1, 1))
		},
		// Floods the bounded channel until canceled.
		func(ctx context.Context, yield func(llm.StreamEvent) bool) {
			for yield(llm.TextDelta{Text: "x"}) {
			}
		},
		events(completed("Clean.", 4, 1)),
	}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1")
	h.waitText("AI > Partial")
	h.key(tcell.KeyCtrlC)
	h.waitText("Cancelled. The partial answer above was not saved.")
	if h.status() != "CANCELLED | Partial response not saved" {
		t.Errorf("status %q", h.status())
	}
	h.waitFor("first worker finished", func(string) bool { _, n := fake.state(); return n == 1 })

	h.submit("Q2")
	h.waitText("AI > xxx")
	h.key(tcell.KeyCtrlC)
	h.waitStatus("CANCELLED")
	h.waitFor("flooding worker finished", func(string) bool { _, n := fake.state(); return n == 2 })

	h.submit("Q3")
	h.waitText("AI > Clean.")
	if strings.Contains(h.text(), "LATE") {
		t.Errorf("a late event was shown:\n%s", h.text())
	}
	sent, _ := fake.state()
	if len(sent[2]) != 1 || sent[2][0].Content != "Q3" {
		t.Errorf("third context %v", sent[2])
	}
	if convs := h.saved(); len(convs) != 1 || len(convs[0].Messages) != 2 || convs[0].Messages[1].Content != "Clean." {
		t.Errorf("saved %+v", convs)
	}

	// Idle Ctrl+C exits with 130.
	h.key(tcell.KeyCtrlC)
	if code := h.exitCode(); code != ExitInterrupted {
		t.Errorf("code %d", code)
	}
}

func TestCtrlCBeforeText(t *testing.T) {
	gate := make(chan struct{})
	fake := &fakeProvider{turns: []turnScript{gated(gate, nil)}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1")
	h.waitStatus("CONNECTING | ")
	h.interrupts <- os.Interrupt // SIGINT works like the Ctrl+C key
	h.waitText("| Cancelled.")
	if h.status() != "CANCELLED" || strings.Contains(h.text(), "partial") {
		t.Errorf("screen:\n%s", h.text())
	}
	h.waitFor("worker finished", func(string) bool { _, n := fake.state(); return n == 1 })
	if convs := h.saved(); len(convs) != 0 {
		t.Errorf("saved %+v", convs)
	}
	h.interrupts <- os.Interrupt
	if code := h.exitCode(); code != ExitInterrupted {
		t.Errorf("code %d", code)
	}
}

// SIGTERM and SIGHUP end the UI: the running request is cancelled and
// not saved, the terminal is restored once, and the code is 128 + signal.
func TestTerminationSignals(t *testing.T) {
	for _, tt := range []struct {
		sig  syscall.Signal
		code int
	}{{syscall.SIGTERM, 143}, {syscall.SIGHUP, 129}} {
		t.Run(tt.sig.String(), func(t *testing.T) {
			gate := make(chan struct{})
			defer close(gate)
			fake := &fakeProvider{turns: []turnScript{gated(gate, deltas("Partial"), completed("Partial answer", 1, 1))}}
			h := start(t, fake, setup{streamed: true, transcript: true})
			h.submit("Q1")
			h.waitText("AI > Partial")
			h.terminations <- tt.sig
			if code := h.exitCode(); code != tt.code {
				t.Errorf("code %d, want %d", code, tt.code)
			}
			h.waitFor("worker finished", func(string) bool { _, n := fake.state(); return n == 1 })
			if _, finis := h.screen.counts(); finis != 1 {
				t.Errorf("terminal finalized %d times", finis)
			}
			if convs := h.saved(); len(convs) != 0 {
				t.Errorf("the cancelled answer was saved: %+v", convs)
			}
			if out := h.stdout.String(); out != "" {
				t.Errorf("transcript %q: nothing was committed", out)
			}
		})
	}
}

// A canceled stream whose worker does not finish refuses the next
// message for a while instead of overlapping it.
func TestCancelledStreamStillClosing(t *testing.T) {
	stuck := make(chan struct{})
	fake := &fakeProvider{turns: []turnScript{
		func(ctx context.Context, yield func(llm.StreamEvent) bool) {
			yield(llm.TextDelta{Text: "Part"})
			<-stuck // a read that ignores the cancellation
		},
		events(completed("Next.", 1, 1)),
	}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1")
	h.waitText("AI > Part")
	h.key(tcell.KeyCtrlC)
	h.waitStatus("CANCELLED")
	h.submit("Q2")
	h.waitStatus("WARNING | The cancelled response is still closing; try again shortly")
	if !strings.Contains(h.text(), "YOU > Q2") || strings.Count(h.text(), "YOU > Q2") != 1 {
		t.Errorf("typed text not kept:\n%s", h.text())
	}
	close(stuck)
	h.key(tcell.KeyEnter)
	h.waitText("AI > Next.")
	if sent, _ := fake.state(); len(sent) != 2 || len(sent[1]) != 1 {
		t.Errorf("sent %v", sent)
	}
}

func TestOutcomes(t *testing.T) {
	fake := &fakeProvider{turns: []turnScript{
		events(completed("", 2, 0)),
		events(append(deltas("Half"), llm.StreamIncomplete{Reason: "the content filter stopped the response", PartialText: "Half",
			Usage: &llm.Usage{InputTokens: intp(5), OutputTokens: intp(1)}})...),
		events(append(deltas("Broken"), llm.StreamFailed{Err: &llm.Error{Kind: llm.KindConnection,
			Message: "Ollama stream ended before the response was complete."}, PartialText: "Broken"})...),
		events(llm.StreamFailed{Err: &llm.Error{Kind: llm.KindAPI, Message: "model crashed"}}),
		events(deltas("No terminal")...),
		events(llm.TextDelta{Text: "Done"}, completed("Done", 1, 1), llm.TextDelta{Text: " AFTER"}, completed("again", 1, 1)),
		func(ctx context.Context, yield func(llm.StreamEvent) bool) { panic("provider bug") },
	}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1")
	h.waitStatus("DONE")
	if !strings.Contains(h.text(), "| AI >"+strings.Repeat(" ", 52)+"|") {
		t.Errorf("empty answer has no entry:\n%s", h.text())
	}
	h.submit("Q2")
	h.waitText("Incomplete: the content filter stopped the response. The partial answer above was not saved.")
	h.submit("Q3")
	h.waitText("Error: Ollama stream ended before the response was complete. The partial answer above was not saved.")
	h.submit("Q4")
	h.waitText("Error: model crashed")
	if strings.Contains(h.text(), "model crashed The partial") {
		t.Error("suffix without text")
	}
	h.submit("Q5")
	h.waitText("Error: The response stream ended without a result. The partial answer above was not saved.")
	h.submit("Q6")
	h.waitText("AI > Done")
	h.waitStatus("DONE")
	h.submit("Q7")
	h.waitText("Error: Unexpected error while streaming: provider bug")
	if strings.Contains(h.text(), "AFTER") {
		t.Error("an event after the terminal one was shown")
	}

	convs := h.saved()
	if len(convs) != 1 {
		t.Fatalf("saved %+v", convs)
	}
	var msgs []string
	for _, m := range convs[0].Messages {
		msgs = append(msgs, m.Content)
	}
	if !reflect.DeepEqual(msgs, []string{"Q1", "", "Q6", "Done"}) {
		t.Errorf("messages %q", msgs)
	}
	var statuses []string
	for _, r := range convs[0].Requests {
		statuses = append(statuses, r.Status)
	}
	if !reflect.DeepEqual(statuses, []string{"completed", "incomplete", "completed"}) {
		t.Errorf("requests %q", statuses)
	}
}

func TestSaveFailureIsAWarning(t *testing.T) {
	fake := &fakeProvider{turns: []turnScript{events(completed("A1", 1, 1)), events(completed("A2", 1, 1))}}
	h := start(t, fake, setup{streamed: true, afterOpen: func(dir string) {
		os.MkdirAll(filepath.Dir(dir), 0o700)
		os.WriteFile(dir, []byte("not a directory"), 0o600) // every save fails
	}})
	h.submit("Q1")
	h.waitText("AI > A1")
	h.waitFor("save warning", func(s string) bool { return strings.Contains(s, "| Warning: ") })
	h.waitStatus("DONE")
	h.key(tcell.KeyLeft)
	h.waitStatus("WARNING | Conversation not saved")
	h.submit("Q2") // the exchange stays in memory
	h.waitText("AI > A2")
	if sent, _ := fake.state(); len(sent[1]) != 3 {
		t.Errorf("context %v", sent[1])
	}
}

func TestCommands(t *testing.T) {
	fake := &fakeProvider{turns: []turnScript{events(completed("A1", 7, 2)), events(completed("A2", 1, 1))}}
	h := start(t, fake, setup{streamed: true})
	h.submit("/HELP")
	h.waitText("In menus: Up/Down move, Enter selects, Esc cancels.")
	h.submit(`\usage`)
	h.waitText("Token usage: no API requests in this conversation yet.")
	h.submit("/nope")
	h.waitText("Unknown command: /nope. Type /help for commands.")
	h.submit("/cleanup")
	h.waitText("No saved conversations.")

	h.submit("Q1")
	h.waitText("AI > A1")
	h.submit("/usage")
	h.waitText("Conversation usage (1 request(s), as reported by the API):")
	h.submit("/reset")
	h.waitText("Started a new conversation. The previous one stays saved.")
	if strings.Contains(h.text(), "AI > A1") || h.status() != "READY | New conversation started" || !strings.Contains(h.text(), " -") {
		t.Errorf("reset:\n%s", h.text())
	}

	h.submit("Q2")
	h.waitText("AI > A2")
	if sent, _ := fake.state(); len(sent[1]) != 1 {
		t.Errorf("context after reset %v", sent[1])
	}
	h.submit("/cleanup")
	h.waitText("Delete 2 saved conversation(s) in /")
	h.waitStatus("WARNING | Type yes to delete saved conversations; anything else cancels")
	h.submit("/exit") // the answer, not a command
	h.waitText("Cleanup cancelled. Nothing was deleted.")
	if len(h.saved()) != 2 {
		t.Error("files deleted")
	}
	h.submit("/cleanup")
	h.waitStatus("WARNING | Type yes")
	h.submit("  YES ")
	h.waitText("Deleted 2 saved conversation(s). Started a new conversation.")
	if len(h.saved()) != 0 || strings.Contains(h.text(), "AI > A2") || h.status() != "READY | Saved conversations deleted" {
		t.Errorf("cleanup:\n%s", h.text())
	}
	if sent, _ := fake.state(); len(sent) != 2 {
		t.Errorf("a command or answer was sent: %v", sent)
	}
	h.submit("/exit")
	if code := h.exitCode(); code != ExitOK {
		t.Errorf("code %d", code)
	}
}

func TestNoHistory(t *testing.T) {
	fake := &fakeProvider{turns: []turnScript{events(completed("A1", 1, 1))}}
	h := start(t, fake, setup{streamed: true, noHistory: true})
	h.submit("Q1")
	h.waitText("AI > A1")
	h.waitStatus("DONE")
	h.key(tcell.KeyLeft)
	h.waitStatus("READY")
	if h.status() != "READY" {
		t.Errorf("status %q", h.status())
	}
	h.submit("/cleanup")
	h.waitText("Conversation history is disabled for this session (--no-history).")
	h.submit("/reset")
	h.waitText("Conversation history cleared.")
	if _, err := os.Stat(h.historyDir); !os.IsNotExist(err) {
		t.Errorf("history directory touched: %v", err)
	}
}

func TestWhileGenerating(t *testing.T) {
	gate := make(chan struct{})
	fake := &fakeProvider{turns: []turnScript{gated(gate, deltas("Working"), completed("Working.", 1, 1)), gated(gate, deltas("Again"))}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1")
	h.waitText("AI > Working")

	h.submit("Another message") // not sent; the text is kept
	h.waitStatus("WARNING | A response is being generated; wait or press Ctrl+C")
	if !strings.Contains(h.text(), "YOU > Another message") {
		t.Errorf("typed text lost:\n%s", h.text())
	}
	h.key(tcell.KeyF4)
	h.waitStatus("WARNING | Settings cannot change while a response is generated")
	h.key(tcell.KeyCtrlD) // never exits during a response, even with empty input
	h.key(tcell.KeyCtrlU)
	h.key(tcell.KeyCtrlD)
	h.key(tcell.KeyF1)
	h.waitText("In menus: Up/Down move, Enter selects, Esc cancels.")
	h.key(tcell.KeyEsc) // nothing
	h.submit("/usage")  // its answer shows that the keys before it were handled
	h.waitText("Token usage: no API requests")
	close(gate)
	h.waitStatus("DONE")
	if sent, _ := fake.state(); len(sent) != 1 {
		t.Errorf("sent %d requests", len(sent))
	}

	h.submit("Q2")
	h.waitText("AI > Again")
	h.submit("/exit") // cancels and exits
	if code := h.exitCode(); code != ExitOK {
		t.Errorf("code %d", code)
	}
	if convs := h.saved(); len(convs) != 1 || len(convs[0].Messages) != 2 {
		t.Errorf("saved %+v", convs)
	}
}

func TestKeys(t *testing.T) {
	h := start(t, &fakeProvider{}, setup{})
	h.key(tcell.KeyEsc) // nothing outside a menu
	h.typeText("x")
	h.key(tcell.KeyCtrlD) // input not empty: no exit
	h.key(tcell.KeyBackspace2)
	h.waitFor("empty input", func(string) bool { return h.row(22) == "| YOU >"+strings.Repeat(" ", 72)+"|" })
	h.key(tcell.KeyCtrlD)
	if code := h.exitCode(); code != ExitOK {
		t.Errorf("code %d", code)
	}
}

func TestInputOnScreen(t *testing.T) {
	h := start(t, &fakeProvider{}, setup{cols: 40, rows: 12})
	h.typeText("Grüß Gott! Привет, мир! 0123456789")
	// Field: 40-4-6 = 30 cells; the text (34 runes) scrolls left.
	h.waitText("| YOU > Gott! Привет, мир! 0123456789  |") // the cursor needs the last cell
	if x, y, _ := h.screen.GetCursor(); x != 37 || y != 9 {
		t.Errorf("cursor %d,%d", x, y)
	}
	h.key(tcell.KeyHome)
	h.waitText("| YOU > Grüß Gott! Привет, мир! 012345 |")
	if x, _, _ := h.screen.GetCursor(); x != 8 {
		t.Errorf("cursor %d", x)
	}
	for range 5 {
		h.key(tcell.KeyRight)
	}
	h.key(tcell.KeyDelete)
	h.key(tcell.KeyBackspace)
	h.key(tcell.KeyCtrlE)
	h.key(tcell.KeyLeft)
	h.key(tcell.KeyCtrlA)
	h.key(tcell.KeyRight)
	h.waitFor("edited", func(s string) bool {
		x, _, _ := h.screen.GetCursor()
		return x == 9 && strings.Contains(s, "| YOU > Grüßott! Привет") // " G" deleted
	})
	h.key(tcell.KeyCtrlU)
	h.waitText("| YOU >                                |")
}

func TestScrolling(t *testing.T) {
	gate := make(chan struct{})
	long := strings.Repeat("line\n", 60) + "end"
	fake := &fakeProvider{turns: []turnScript{
		events(completed(long, 1, 1)),
		gated(gate, deltas("New"), append(deltas("\nmore\nmore\nmore"), completed("New\nmore\nmore\nmore", 1, 1))...),
	}}
	h := start(t, fake, setup{streamed: true})
	h.submit("Q1") // 1 + 1 + 61 lines; 18 rows: at most 45 up
	h.waitText("end")
	h.key(tcell.KeyPgUp)
	h.waitText("scrolled up 17 (PgDn)")
	h.key(tcell.KeyUp)
	h.waitText("scrolled up 18 (PgDn)")
	h.key(tcell.KeyDown)
	h.waitText("scrolled up 17 (PgDn)")
	for range 5 {
		h.key(tcell.KeyPgUp)
	}
	h.waitText("scrolled up 45 (PgDn)") // the oldest line is at the top
	if h.row(1) != "| YOU > Q1"+strings.Repeat(" ", 48)+"| F2 MODEL           |" {
		t.Errorf("top row %q", h.row(1))
	}

	// Submitting returns to the bottom; the view follows new text there.
	h.submit("Q2")
	h.waitText("AI > New")
	if strings.Contains(h.row(0), "scrolled") {
		t.Error("still scrolled after submitting")
	}
	// Scrolled up, incoming text does not move the visible lines.
	h.key(tcell.KeyPgUp)
	h.waitText("scrolled up 17 (PgDn)")
	conversationRows := func() []string {
		var rows []string
		for _, r := range h.rows()[1:19] {
			rows = append(rows, string([]rune(r)[:58]))
		}
		return rows
	}
	before := conversationRows()
	close(gate)
	h.waitStatus("DONE")
	h.waitText("scrolled up 20 (PgDn)")
	if after := conversationRows(); !reflect.DeepEqual(before, after) {
		t.Errorf("the view moved:\n%s\n---\n%s", strings.Join(before, "\n"), strings.Join(after, "\n"))
	}

	// A resize keeps the offset from the bottom.
	h.resize(70, 15)
	h.waitText("scrolled up 20 (PgDn)")
	for range 3 {
		h.key(tcell.KeyPgDn)
	}
	h.waitFor("bottom", func(string) bool { return !strings.Contains(h.row(0), "scrolled") })
	if !strings.Contains(h.text(), "| AI > New") {
		t.Errorf("screen:\n%s", h.text())
	}
}
