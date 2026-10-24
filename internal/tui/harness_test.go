package tui

import (
	"bytes"
	"context"
	"errors"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gdamore/tcell/v2"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/session"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// turnScript produces the events of one request. It must stop when yield
// returns false and should watch ctx like a real provider.
type turnScript func(ctx context.Context, yield func(llm.StreamEvent) bool)

// fakeProvider is an llm.StreamingProvider that replays one script per
// request, streamed or as a complete answer, and records what was sent
// and how many requests finished. Its discovery answers are configurable;
// gate, if set, holds every discovery call until it is closed.
type fakeProvider struct {
	mu       sync.Mutex
	turns    []turnScript
	sent     [][]llm.Message
	requests []llm.Request
	finished int

	name         string // default "ollama"
	models       []llm.Model
	listErr      error
	defaultModel string
	unavailable  error
	support      map[string]llm.ReasoningSupport // default: Ollama off/on
	supportErr   error
	limits       map[string]int
	gate         chan struct{}
	calls        map[string]int // discovery calls by method
}

func (f *fakeProvider) next(req llm.Request) turnScript {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sent = append(f.sent, req.Messages)
	f.requests = append(f.requests, req)
	script := f.turns[0]
	f.turns = f.turns[1:]
	return script
}

func (f *fakeProvider) done() {
	f.mu.Lock()
	f.finished++
	f.mu.Unlock()
}

func (f *fakeProvider) Stream(ctx context.Context, req llm.Request) iter.Seq[llm.StreamEvent] {
	return func(yield func(llm.StreamEvent) bool) {
		script := f.next(req)
		defer f.done()
		script(ctx, yield)
	}
}

// Chat runs the script and returns its terminal event as an answer.
func (f *fakeProvider) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	script := f.next(req)
	defer f.done()
	var resp llm.Response
	err := errors.New("no terminal event")
	script(ctx, func(e llm.StreamEvent) bool {
		switch e := e.(type) {
		case llm.StreamCompleted:
			resp, err = e.Response, nil
		case llm.StreamIncomplete:
			resp, err = llm.Response{Text: e.PartialText, IncompleteReason: e.Reason, Usage: e.Usage}, nil
		case llm.StreamFailed:
			err = e.Err
		}
		return true
	})
	return resp, err
}

func (f *fakeProvider) Name() string {
	if f.name == "" {
		return "ollama"
	}
	return f.name
}

// discovery records a call and waits for the gate.
func (f *fakeProvider) discovery(ctx context.Context, method string) error {
	f.mu.Lock()
	if f.calls == nil {
		f.calls = map[string]int{}
	}
	f.calls[method]++
	gate := f.gate
	f.mu.Unlock()
	if gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func (f *fakeProvider) callCount(method string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[method]
}

func (f *fakeProvider) ListModels(ctx context.Context) ([]llm.Model, error) {
	if err := f.discovery(ctx, "ListModels"); err != nil {
		return nil, err
	}
	return f.models, f.listErr
}

func (f *fakeProvider) DefaultModel(ctx context.Context) (string, error) {
	if err := f.discovery(ctx, "DefaultModel"); err != nil {
		return "", err
	}
	return f.defaultModel, nil
}

func (f *fakeProvider) CheckAvailability(ctx context.Context) error {
	if err := f.discovery(ctx, "CheckAvailability"); err != nil {
		return err
	}
	return f.unavailable
}

var onOff = llm.ReasoningSupport{Values: []string{"off", "on"}, Known: true, Default: "off", NoModelDefault: true}

func (f *fakeProvider) ReasoningSupport(ctx context.Context, model string) (llm.ReasoningSupport, error) {
	if err := f.discovery(ctx, "ReasoningSupport"); err != nil {
		return llm.ReasoningSupport{}, err
	}
	if f.supportErr != nil {
		return llm.ReasoningSupport{}, f.supportErr
	}
	if s, ok := f.support[model]; ok {
		return s, nil
	}
	return onOff, nil
}

func (f *fakeProvider) MaxOutputLimit(ctx context.Context, model string) (int, error) {
	if err := f.discovery(ctx, "MaxOutputLimit"); err != nil {
		return 0, err
	}
	return f.limits[model], nil
}

func (f *fakeProvider) lastRequest() llm.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[len(f.requests)-1]
}

func (f *fakeProvider) state() ([][]llm.Message, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.sent, f.finished
}

// events yields the events in order.
func events(evs ...llm.StreamEvent) turnScript {
	return func(ctx context.Context, yield func(llm.StreamEvent) bool) {
		for _, e := range evs {
			if !yield(e) {
				return
			}
		}
	}
}

// gated yields before, waits for gate (or cancellation), then yields after.
func gated(gate <-chan struct{}, before []llm.StreamEvent, after ...llm.StreamEvent) turnScript {
	return func(ctx context.Context, yield func(llm.StreamEvent) bool) {
		for _, e := range before {
			if !yield(e) {
				return
			}
		}
		select {
		case <-gate:
		case <-ctx.Done():
			yield(llm.StreamFailed{Err: ctx.Err()})
			return
		}
		for _, e := range after {
			if !yield(e) {
				return
			}
		}
	}
}

func intp(n int) *int { return &n }

func completed(text string, input, output int) llm.StreamCompleted {
	return llm.StreamCompleted{Response: llm.Response{Text: text,
		Usage: &llm.Usage{InputTokens: intp(input), OutputTokens: intp(output), TotalTokens: intp(input + output)}}}
}

func deltas(texts ...string) []llm.StreamEvent {
	var evs []llm.StreamEvent
	for _, t := range texts {
		evs = append(evs, llm.TextDelta{Text: t})
	}
	return evs
}

// syncBuffer is a bytes.Buffer safe for concurrent use.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
	log *[]string // shared order of writes and Fini, if set
	tag string
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.log != nil {
		*b.log = append(*b.log, b.tag)
	}
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// testScreen is a simulation screen that records the cells written and
// Fini, and can panic on a redraw.
type testScreen struct {
	tcell.SimulationScreen
	// drawMu orders Show with snapshots: GetContents returns the cells
	// themselves, not a copy.
	drawMu     sync.Mutex
	mu         sync.Mutex
	bottomHits int // writes to the bottom-right cell
	finis      int
	shows      int
	armed      bool // the next Show panics
	log        *[]string
}

func (s *testScreen) SetContent(x, y int, primary rune, combining []rune, style tcell.Style) {
	w, h := s.SimulationScreen.Size()
	if x == w-1 && y == h-1 {
		s.mu.Lock()
		s.bottomHits++
		s.mu.Unlock()
	}
	s.SimulationScreen.SetContent(x, y, primary, combining, style)
}

func (s *testScreen) Show() {
	s.mu.Lock()
	s.shows++
	boom := s.armed
	s.mu.Unlock()
	if boom {
		panic("simulated render failure")
	}
	s.drawMu.Lock()
	defer s.drawMu.Unlock()
	s.SimulationScreen.Show()
}

func (s *testScreen) Sync() {
	s.drawMu.Lock()
	defer s.drawMu.Unlock()
	s.SimulationScreen.Sync()
}

// snapshot returns a copy of the cells shown.
func (s *testScreen) snapshot() ([]tcell.SimCell, int, int) {
	s.drawMu.Lock()
	defer s.drawMu.Unlock()
	cells, w, h := s.GetContents()
	copied := make([]tcell.SimCell, len(cells))
	for i, c := range cells {
		copied[i] = tcell.SimCell{Style: c.Style, Runes: append([]rune(nil), c.Runes...)}
	}
	return copied, w, h
}

func (s *testScreen) setSize(w, h int) {
	s.drawMu.Lock()
	defer s.drawMu.Unlock()
	s.SetSize(w, h)
}

func (s *testScreen) Fini() {
	s.mu.Lock()
	s.finis++
	if s.log != nil {
		*s.log = append(*s.log, "fini")
	}
	s.mu.Unlock()
	s.SimulationScreen.Fini()
}

// armPanic makes the next redraw panic, like a bug in the UI.
func (s *testScreen) armPanic() {
	s.mu.Lock()
	s.armed = true
	s.mu.Unlock()
}

func (s *testScreen) counts() (bottom, finis int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bottomHits, s.finis
}

// harness runs the app on a simulated 80x25 screen with a fake provider
// and a session in a temporary history directory.
type harness struct {
	t            *testing.T
	screen       *testScreen
	fake         *fakeProvider
	sess         *session.Session
	historyDir   string
	settingsPath string
	engine       *chat.Engine
	stdout       *syncBuffer
	stderr       *syncBuffer
	interrupts   chan os.Signal
	terminations chan os.Signal
	cancel       context.CancelFunc
	done         chan int
	log          []string
}

type setup struct {
	streamed   bool
	noHistory  bool
	cols       int
	rows       int
	title      string
	model      string        // default "qwen3.5:4b"
	settings   chat.Settings // default: answer style normal
	reasoning  string        // displayed reasoning; default "off"
	overrides  chat.Overrides
	warnings   []string
	transcript bool
	// others are the providers F3 can switch to, by name.
	others map[string]*fakeProvider
	// settingsJSON, if set, is written to settings.json before the start.
	settingsJSON string
	// settingsDir replaces config/neurobrau as the settings directory.
	settingsDir string
	// prepare runs before the session is opened (e.g. to save files).
	prepare func(historyDir string)
	// afterOpen runs after the session is opened.
	afterOpen func(historyDir string)
}

func start(t *testing.T, fake *fakeProvider, st setup) *harness {
	t.Helper()
	h := &harness{t: t, fake: fake, interrupts: make(chan os.Signal, 1),
		terminations: make(chan os.Signal, 1), done: make(chan int, 1)}
	dir := t.TempDir()
	h.historyDir = filepath.Join(dir, "state", "neurobrau", "conversations")
	if st.settingsDir == "" {
		st.settingsDir = filepath.Join("config", "neurobrau")
	}
	h.settingsPath = filepath.Join(dir, st.settingsDir, "settings.json")
	h.stdout = &syncBuffer{log: &h.log, tag: "stdout"}
	h.stderr = &syncBuffer{log: &h.log, tag: "stderr"}
	if st.prepare != nil {
		st.prepare(h.historyDir)
	}
	if st.settingsJSON != "" {
		os.MkdirAll(filepath.Dir(h.settingsPath), 0o700)
		os.WriteFile(h.settingsPath, []byte(st.settingsJSON), 0o600)
	}
	var store *storage.HistoryStore
	if !st.noHistory {
		store = storage.NewHistoryStore(h.historyDir)
	}
	if st.model == "" {
		st.model = "qwen3.5:4b"
	}
	sess, _ := session.Open(session.Options{Store: store, Provider: fake.Name(), Model: st.model})
	sess.SetDefaults(fake.Name(), st.model)
	h.sess = sess
	if st.afterOpen != nil {
		st.afterOpen(h.historyDir)
	}

	sim := tcell.NewSimulationScreen("UTF-8")
	if err := sim.Init(); err != nil {
		t.Fatal(err)
	}
	if st.cols == 0 {
		st.cols, st.rows = 80, 25
	}
	sim.SetSize(st.cols, st.rows)
	h.screen = &testScreen{SimulationScreen: sim, log: &h.log}
	if st.title == "" {
		st.title = "NeuroBräu Go CLI test"
	}
	if st.settings.AnswerStyle == "" {
		st.settings.AnswerStyle = llm.StyleNormal
	}
	if st.reasoning == "" {
		st.reasoning = "off"
	}
	settingsStore := storage.NewSettingsStore(h.settingsPath)
	saved, _ := settingsStore.Load()
	h.engine = &chat.Engine{Provider: fake, Model: st.model, Settings: st.settings}
	cfg := Config{
		Screen: h.screen, Title: st.title, Session: sess, Engine: h.engine, Stream: st.streamed,
		Reasoning:   st.reasoning,
		Preferences: Preferences{Store: settingsStore, Saved: saved, Overrides: st.overrides},
		NewProvider: func(name string) (llm.StreamingProvider, error) {
			if p, ok := st.others[name]; ok {
				return p, nil
			}
			return nil, errors.New("not configured in this test")
		},
		Warnings: st.warnings, TranscriptOnExit: st.transcript,
		Stdout: h.stdout, Stderr: h.stderr, Interrupts: h.interrupts,
		Terminations: h.terminations,
	}
	ctx, cancel := context.WithCancel(context.Background())
	h.cancel = cancel
	go func() { h.done <- Run(ctx, cfg) }()
	t.Cleanup(func() {
		cancel() // ends a still running app with 130
		select {
		case <-h.done:
		case <-time.After(5 * time.Second):
			t.Error("the app did not exit")
		}
	})
	h.waitFor("first frame", func(s string) bool { return strings.Contains(s, "YOU >") })
	return h
}

func (h *harness) key(k tcell.Key) { h.screen.InjectKey(k, 0, tcell.ModNone) }

func (h *harness) typeText(text string) {
	for _, r := range text {
		h.screen.InjectKey(tcell.KeyRune, r, tcell.ModNone)
	}
}

// submit types text and presses Enter.
func (h *harness) submit(text string) {
	h.typeText(text)
	h.key(tcell.KeyEnter)
}

func (h *harness) resize(cols, rows int) {
	h.screen.setSize(cols, rows)
	h.screen.PostEvent(tcell.NewEventResize(cols, rows))
}

// rows returns the screen as text, one string per row.
func (h *harness) rows() []string {
	cells, w, hgt := h.screen.snapshot()
	rows := make([]string, hgt)
	for y := range hgt {
		var b strings.Builder
		for x := range w {
			c := cells[y*w+x]
			if len(c.Runes) == 0 {
				b.WriteRune(' ')
			} else {
				b.WriteRune(c.Runes[0])
			}
		}
		rows[y] = b.String()
	}
	return rows
}

func (h *harness) text() string { return strings.Join(h.rows(), "\n") }

func (h *harness) row(y int) string { return strings.TrimRight(h.rows()[y], " ") }

func (h *harness) cell(x, y int) (rune, tcell.Style) {
	cells, w, _ := h.screen.snapshot()
	c := cells[y*w+x]
	return c.Runes[0], c.Style
}

// waitFor waits until cond holds for the screen text.
func (h *harness) waitFor(what string, cond func(screen string) bool) {
	h.t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		s := h.text()
		if cond(s) {
			return
		}
		if time.Now().After(deadline) {
			h.t.Fatalf("timed out waiting for %s; screen:\n%s", what, s)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// waitText waits until text is on the screen, either as is or in the
// conversation with its wrapping undone.
func (h *harness) waitText(text string) {
	h.t.Helper()
	h.waitFor(`"`+text+`"`, func(s string) bool {
		return strings.Contains(s, text) || strings.Contains(h.conversation(), strings.Join(strings.Fields(text), " "))
	})
}

// conversation is the visible conversation text with lines joined and
// runs of spaces collapsed.
func (h *harness) conversation() string {
	l, ok := NewLayout(h.screen.Size())
	if !ok {
		return ""
	}
	var words []string
	for _, row := range h.rows()[conversationY:l.StatusSeparatorRow()] {
		r := []rune(row)
		words = append(words, strings.Fields(string(r[2:l.ConversationLimit()]))...)
	}
	return strings.Join(words, " ")
}

// at is the rune at a cell.
func (h *harness) at(x, y int) rune {
	r, _ := h.cell(x, y)
	return r
}

// status is the runtime status row without its borders.
func (h *harness) status() string {
	l, _ := NewLayout(h.screen.Size())
	return strings.TrimSpace(strings.Trim(h.row(l.StatusRow()), "|"))
}

func (h *harness) waitStatus(prefix string) {
	h.t.Helper()
	h.waitFor("status "+prefix, func(string) bool { return strings.HasPrefix(h.status(), prefix) })
}

// exitCode waits for the app to exit.
func (h *harness) exitCode() int {
	h.t.Helper()
	select {
	case code := <-h.done:
		h.done <- code // for the cleanup
		return code
	case <-time.After(5 * time.Second):
		h.t.Fatalf("the app did not exit; screen:\n%s", h.text())
		return -1
	}
}

// saved returns the conversations saved in the history directory.
func (h *harness) saved() []storage.Conversation {
	h.t.Helper()
	files, _ := filepath.Glob(filepath.Join(h.historyDir, "*.json"))
	var convs []storage.Conversation
	for _, f := range files {
		if filepath.Base(f) == "active.json" {
			continue
		}
		c, err := storage.NewHistoryStore(h.historyDir).Load(f)
		if err != nil {
			h.t.Fatal(err)
		}
		convs = append(convs, c)
	}
	return convs
}
