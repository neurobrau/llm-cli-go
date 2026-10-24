package tui

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"syscall"
	"time"
	"unicode"

	"github.com/gdamore/tcell/v2"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/session"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// Exit codes (docs/tui-spec.md, sections 9 and 16).
const (
	ExitOK          = 0
	ExitError       = 1
	ExitInterrupted = 130
)

// Timing of the event loop (docs/tui-spec.md, sections 5 and 14).
const (
	tickInterval    = 100 * time.Millisecond // elapsed-time redraw while a request runs
	eventsPerRedraw = 256                    // stream events applied between two redraws
	closeWait       = time.Second            // a new request waits this long for a canceled one to close
	exitWait        = 500 * time.Millisecond // exit waits this long for open requests
	flashDuration   = 2 * time.Second        // a warning during a request outranks the live status
)

// HelpLine is the key hint on the last row.
const HelpLine = " F1 /help | /reset /exit | F2-F6 settings | PgUp/PgDn scroll | Ctrl+C quit"

// HelpText is the F1 and /help text of the fullscreen UI.
const HelpText = `Commands:
  /help     show this help
  /reset    start a new conversation (the previous one stays saved)
  /cleanup  delete all saved conversations (asks for confirmation)
  /usage    token usage and ESTIMATED cost of this conversation
  /exit     exit the application (the conversation stays saved)
Backslash aliases also work: \help, \reset, \cleanup, \usage, \exit.
Keys: F1 this help, Enter send, Backspace/Delete edit, Left/Right/Home/End move,
Ctrl+U clear input, PgUp/PgDn/Up/Down scroll, Ctrl+D (empty input) or Ctrl+C quit.
Ctrl+C while a response is generated cancels it.
F2 model, F3 provider (both switch to that model's own conversation).
F4 reasoning, F5 max output tokens, F6 answer style (keep the conversation;
they apply from the next message and are saved per model).
In menus: Up/Down move, Enter selects, Esc cancels.`

// Config is what Run needs. All fields but Warnings, TranscriptOnExit,
// Interrupts, Terminations and Now are required.
type Config struct {
	// Screen is initialized by the caller; Run finalizes it (restoring the
	// terminal) on every exit path.
	Screen tcell.Screen
	// Title is shown in the top border, e.g. "NeuroBräu Go CLI 0.4.0-dev".
	Title   string
	Session *session.Session
	// Engine sends the requests. F2–F6 change its provider, model and
	// settings, only while no request or check runs.
	Engine *chat.Engine
	// Stream shows answers while they are generated (--stream).
	Stream bool
	// Reasoning is the effective reasoning of Engine.Settings, for display
	// ("" sends no parameter).
	Reasoning   string
	Preferences Preferences
	// NewProvider creates a provider by name, without a request (F3).
	NewProvider func(name string) (llm.StreamingProvider, error)
	// Warnings are startup warnings, shown as error entries.
	Warnings []string
	// TranscriptOnExit prints the committed messages to Stdout after the
	// terminal is restored.
	TranscriptOnExit bool
	Stdout, Stderr   io.Writer
	// Interrupts delivers SIGINT (e.g. `kill -INT`); in the fullscreen UI
	// Ctrl+C arrives as a key. Nil: never.
	Interrupts <-chan os.Signal
	// Terminations delivers SIGTERM and SIGHUP: the running request is
	// cancelled, the terminal restored, and Run returns 128 + the signal
	// number. Nil: never.
	Terminations <-chan os.Signal
	// Now is the clock; nil: time.Now.
	Now func() time.Time
}

// Preferences are the saved settings and this run's command-line
// overrides. A confirmed F4–F6 choice replaces the override for the rest
// of the run and is saved.
type Preferences struct {
	// Store writes settings.json; nil: the location is unknown. A store
	// whose file could not be read refuses to write.
	Store     *storage.SettingsStore
	Saved     storage.Settings
	Overrides chat.Overrides
}

// request is a running request. Only the event loop reads or changes it.
// Its events channel is its generation token: once the request is retired
// (finished or canceled) the loop no longer applies anything from it.
type request struct {
	text      string // the user message
	streamed  bool
	cancel    context.CancelFunc
	events    <-chan llm.StreamEvent // the worker's bounded hand-off; closed when it is done
	started   time.Time
	firstText time.Time // zero: no text yet
	answer    string
	thinking  bool
	entry     *entry // the AI entry, nil until text arrives
}

type app struct {
	Config
	ctx    context.Context
	styles styles

	transcript     transcript
	input          inputLine
	status         status
	afterTransient *status // shown when a transient status clears
	flash          *status // shown over the live status until flashUntil
	flashUntil     time.Time
	confirming     bool // the next submitted line answers the /cleanup prompt

	active  *request
	closing <-chan llm.StreamEvent // the last finished or canceled request, until its worker is done

	op        *operation    // a running check or discovery
	opClosing chan struct{} // a retired operation, until its goroutine is done

	providers   map[string]llm.StreamingProvider // created once, keeping their caches
	reasoning   string                           // effective reasoning, for display
	menu        *menu                            // the open menu
	tokenPrompt bool                             // the input line asks for a custom max tokens value
	tokenLimit  int                              // known limit for the custom value (0: unknown)
	draft       inputLine                        // the typed text, set aside during the token prompt

	exiting  bool
	exitCode int
}

// Run runs the fullscreen chat until /exit, Ctrl+D on empty input (0),
// Ctrl+C while idle (130) or a screen failure (1), and returns the exit
// code. Canceling ctx also ends it with 130. The screen is finalized on
// every exit, also after a panic in the UI; with TranscriptOnExit the
// committed messages are printed afterwards.
func Run(ctx context.Context, cfg Config) (code int) {
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	a := &app{Config: cfg, ctx: ctx, styles: newStyles(cfg.Screen.Colors()), status: ready(),
		reasoning: cfg.Reasoning,
		providers: map[string]llm.StreamingProvider{}}
	if p, ok := cfg.Engine.Provider.(llm.StreamingProvider); ok {
		a.providers[p.Name()] = p
	}
	events := make(chan tcell.Event, 64)
	quit := make(chan struct{})
	go cfg.Screen.ChannelEvents(events, quit)
	defer func() {
		r := recover()
		close(quit)
		a.stopRequests()
		cfg.Screen.Fini()
		if r != nil {
			fmt.Fprintf(cfg.Stderr, "neurobrau: error: the fullscreen UI failed: %v\n", r)
			code = ExitError
		}
		if cfg.TranscriptOnExit {
			a.printTranscript()
		}
	}()
	cfg.Screen.SetStyle(a.styles[roleText])
	a.showRestored()
	return a.loop(events)
}

// showRestored shows the restored messages, then the startup warnings
// and the restored-conversation notice.
func (a *app) showRestored() {
	a.showMessages(a.Session.Messages())
	for _, w := range a.Warnings {
		a.addEntry(entryError, "Warning: "+w)
	}
	if notice := chat.RestoredNotice(a.Session); notice != "" {
		a.addEntry(entryInfo, notice)
	}
}

// showMessages adds committed messages as user and AI entries.
func (a *app) showMessages(messages []llm.Message) {
	for _, m := range messages {
		kind := entryAI
		if m.Role == llm.RoleUser {
			kind = entryUser
		}
		a.addEntry(kind, m.Content)
	}
}

// loop is the event loop: it redraws, then waits for a key, a stream
// event, the redraw tick, SIGINT or the end of ctx. All drawing and all
// changes to what the screen shows happen here.
func (a *app) loop(events <-chan tcell.Event) int {
	ticker := time.NewTicker(tickInterval)
	defer ticker.Stop()
	for !a.exiting {
		a.render()
		var tick <-chan time.Time
		var answer <-chan llm.StreamEvent
		if a.active != nil {
			answer = a.active.events
		}
		var opResult <-chan func()
		if a.op != nil {
			opResult = a.op.result
		}
		if a.active != nil || a.flash != nil {
			tick = ticker.C
		}
		select {
		case ev, ok := <-events:
			if !ok {
				fmt.Fprintln(a.Stderr, "neurobrau: error: the terminal stopped responding")
				return ExitError
			}
			a.handleEvent(ev)
		keys: // more pending keys (e.g. pasted text) share one redraw
			for i := 1; i < eventsPerRedraw && !a.exiting; i++ {
				select {
				case ev, ok := <-events:
					if !ok {
						break keys // reported by the next select
					}
					a.handleEvent(ev)
				default:
					break keys
				}
			}
		case event, ok := <-answer:
			a.apply(event, ok)
		queued: // apply queued events, then redraw once
			for i := 1; i < eventsPerRedraw && a.active != nil; i++ {
				select {
				case event, ok := <-a.active.events:
					a.apply(event, ok)
				default:
					break queued
				}
			}
		case apply := <-opResult:
			a.finishOp(apply)
		case _, ok := <-a.closing: // events of a retired request are dropped
			if !ok {
				a.closing = nil
			}
		case <-a.opClosing:
			a.opClosing = nil
		case <-tick:
		case <-a.Interrupts:
			a.interrupt()
		case sig := <-a.Terminations:
			return SignalExitCode(sig)
		case <-a.ctx.Done():
			return ExitInterrupted
		}
	}
	return a.exitCode
}

// SignalExitCode is the shell convention for a run ended by sig: 128 +
// its number (SIGTERM 143, SIGHUP 129).
func SignalExitCode(sig os.Signal) int {
	if s, ok := sig.(syscall.Signal); ok {
		return 128 + int(s)
	}
	return ExitError
}

func (a *app) exit(code int) {
	a.exiting, a.exitCode = true, code
}

func (a *app) now() time.Time { return a.Now() }

// handleEvent applies a key press or a resize.
func (a *app) handleEvent(ev tcell.Event) {
	switch ev := ev.(type) {
	case *tcell.EventResize:
		a.Screen.Sync() // the scroll offset counts from the bottom, so the view stays put
	case *tcell.EventKey:
		a.handleKey(ev)
	}
}

func (a *app) handleKey(ev *tcell.EventKey) {
	if a.status.transient() {
		// DONE, CANCELLED, ERROR and WARNING last until the next key press.
		a.status = ready()
		if a.afterTransient != nil {
			a.status = *a.afterTransient
		}
		a.afterTransient = nil
	}
	if a.menu != nil {
		a.menuKey(ev) // nothing reaches the input line
		return
	}
	switch key := ev.Key(); key {
	case tcell.KeyCtrlC:
		a.interrupt()
	case tcell.KeyF1:
		a.addEntry(entryInfo, HelpText) // also while a response is generated
		a.transcript.scroll = 0
	case tcell.KeyF2, tcell.KeyF3, tcell.KeyF4, tcell.KeyF5, tcell.KeyF6:
		if !a.tokenPrompt { // ignored while the token prompt is open
			a.functionKey(key)
		}
	case tcell.KeyEsc:
		if a.tokenPrompt {
			a.endTokenPrompt()
			a.addEntry(entryInfo, "Max output tokens unchanged.")
		}
	case tcell.KeyEnter, tcell.KeyCtrlJ:
		a.submit()
	case tcell.KeyBackspace, tcell.KeyBackspace2:
		a.input.backspace()
	case tcell.KeyDelete:
		a.input.delete()
	case tcell.KeyLeft:
		a.input.left()
	case tcell.KeyRight:
		a.input.right()
	case tcell.KeyHome, tcell.KeyCtrlA:
		a.input.home()
	case tcell.KeyEnd, tcell.KeyCtrlE:
		a.input.end()
	case tcell.KeyCtrlU:
		a.input.clear()
	case tcell.KeyCtrlD:
		// Exits only on empty input, and never while a response is generated.
		if !a.busy() && len(a.input.text) == 0 {
			a.exit(ExitOK)
		}
	case tcell.KeyUp:
		a.scrollBy(1)
	case tcell.KeyDown:
		a.scrollBy(-1)
	case tcell.KeyPgUp:
		a.scrollBy(a.pageSize())
	case tcell.KeyPgDn:
		a.scrollBy(-a.pageSize())
	case tcell.KeyRune:
		if r := ev.Rune(); unicode.IsPrint(r) {
			a.input.insert(r)
		}
	}
	// Other keys, including Esc outside a menu or prompt, do nothing.
}

// interrupt is Ctrl+C: cancel the running request or check, else exit
// with 130.
func (a *app) interrupt() {
	switch {
	case a.active != nil:
		a.cancelActive()
	case a.op != nil:
		a.cancelOp()
	default:
		a.exit(ExitInterrupted)
	}
}

func (a *app) flashWarning(text string) {
	s := warning(text)
	a.flash, a.flashUntil = &s, a.now().Add(flashDuration)
}

// submit handles the input line as a command or a message
// (docs/tui-spec.md, section 8).
func (a *app) submit() {
	text := strings.TrimSpace(a.input.String())
	if a.busy() {
		a.submitWhileBusy(text)
		return
	}
	a.input.clear()
	if a.tokenPrompt {
		a.submitTokens(text)
		return
	}
	if text == "" {
		return
	}
	switch name, isCommand := chat.ParseCommand(text); {
	case a.confirming:
		a.confirmCleanup(text) // never sent or run as a command
	case isCommand:
		a.command(name)
	default:
		a.send(text)
	}
	a.transcript.scroll = 0
}

// submitWhileBusy: only one request or check runs at a time. /exit cancels
// it and exits; /help and /usage work; anything else keeps the typed text.
func (a *app) submitWhileBusy(text string) {
	switch name, _ := chat.ParseCommand(text); {
	case name == "exit":
		a.exit(ExitOK) // stopRequests cancels it
	case name == "help" || name == "usage":
		a.input.clear()
		a.command(name)
		a.transcript.scroll = 0
	case text != "" && a.active != nil:
		a.flashWarning("A response is being generated; wait or press Ctrl+C")
	case text != "":
		a.flashWarning("Wait for the current check or press Ctrl+C")
	}
}

func (a *app) command(name string) {
	switch name {
	case "exit":
		a.exit(ExitOK)
	case "help":
		a.addEntry(entryInfo, HelpText)
	case "usage":
		a.addEntry(entryInfo, chat.UsageReport(a.Session.Requests()))
	case "reset":
		err := a.Session.Reset()
		a.clearPanel(chat.ResetNotice(a.Session.Persistent()))
		a.status = ready("New conversation started")
		if err != nil {
			a.addEntry(entryError, "Warning: "+err.Error())
			a.status = warning("Conversation not saved")
		}
	case "cleanup":
		a.cleanup()
	default:
		a.addEntry(entryError, fmt.Sprintf("Unknown command: /%s. Type /help for commands.", name))
	}
}

// cleanup asks to confirm deleting the saved conversations; the next
// submitted line is the answer.
func (a *app) cleanup() {
	if !a.Session.Persistent() {
		a.addEntry(entryInfo, chat.CleanupWithoutHistory)
		return
	}
	count, err := a.Session.SavedCount()
	switch {
	case err != nil:
		a.showError(err.Error())
	case count == 0:
		a.addEntry(entryInfo, chat.NoSavedConversations)
	default:
		a.addEntry(entryInfo, chat.CleanupPrompt(count, a.Session.HistoryDir()))
		a.confirming = true
	}
}

func (a *app) confirmCleanup(answer string) {
	a.confirming = false
	if !chat.CleanupConfirmed(answer) {
		a.addEntry(entryInfo, chat.CleanupCancelled)
		return
	}
	deleted, err := a.Session.Cleanup()
	if err != nil {
		a.showError(err.Error())
		return
	}
	a.clearPanel(chat.CleanupDone(deleted))
	a.status = ready("Saved conversations deleted")
}

func (a *app) showError(msg string) {
	a.addEntry(entryError, "Error: "+msg)
	a.status = status{stateError, []string{msg}}
}

func (a *app) clearPanel(notice string) {
	a.transcript.entries = nil
	a.transcript.scroll = 0
	a.addEntry(entryInfo, notice)
}

// addEntry appends an entry to the panel.
func (a *app) addEntry(kind entryKind, text string) {
	a.anchored(func() { a.transcript.entries = append(a.transcript.entries, &entry{kind: kind, text: text}) })
}

// anchored applies a change to the entries. At the bottom the view
// follows new lines; when scrolled up, the offset grows by the lines the
// change added, so the visible lines do not move.
func (a *app) anchored(change func()) {
	l, ok := a.layout()
	if !ok || a.transcript.scroll == 0 {
		change()
		return
	}
	width := l.ConversationWidth()
	before := a.transcript.lineCount(width)
	change()
	a.transcript.scroll += a.transcript.lineCount(width) - before
}

func (a *app) layout() (Layout, bool) {
	return NewLayout(a.Screen.Size())
}

func (a *app) scrollBy(delta int) {
	a.transcript.scroll = max(0, min(a.maxScroll(), a.transcript.scroll+delta))
}

func (a *app) pageSize() int {
	if l, ok := a.layout(); ok {
		return max(1, l.ConversationHeight()-1)
	}
	return 1
}

func (a *app) maxScroll() int {
	l, ok := a.layout()
	if !ok {
		return 0
	}
	return max(0, a.transcript.lineCount(l.ConversationWidth())-l.ConversationHeight())
}

// printTranscript prints the committed messages, separated by blank
// lines (docs/tui-spec.md, section 9). Nothing for an empty conversation.
func (a *app) printTranscript() {
	messages := a.Session.Messages()
	if len(messages) == 0 {
		return
	}
	parts := make([]string, len(messages))
	for i, m := range messages {
		label := aiLabel
		if m.Role == llm.RoleUser {
			label = userLabel
		}
		parts[i] = label + m.Content
	}
	fmt.Fprintln(a.Stdout, strings.Join(parts, "\n\n"))
}
