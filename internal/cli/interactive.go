package cli

import (
	"context"
	"fmt"
	"os"

	"github.com/gdamore/tcell/v2"
	"golang.org/x/term"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/openai"
	"github.com/neurobrau/llm-cli-go/internal/session"
	"github.com/neurobrau/llm-cli-go/internal/storage"
	"github.com/neurobrau/llm-cli-go/internal/tui"
)

// Terminal access; tests replace these.
var (
	// isTerminal reports whether a standard stream is a terminal.
	isTerminal = func(stream any) bool {
		f, ok := stream.(*os.File)
		return ok && term.IsTerminal(int(f.Fd()))
	}
	// newScreen creates the fullscreen UI's screen (not initialized yet).
	newScreen = tcell.NewScreen
)

// fullscreen reports whether the interactive chat uses the fullscreen
// UI: not with --plain, and only when stdin and stdout are terminals and
// TERM is set and not "dumb" (docs/tui-spec.md, section 1).
func fullscreen(opts options, s Streams) bool {
	termName := os.Getenv("TERM")
	return !opts.plain && termName != "" && termName != "dumb" && isTerminal(s.Stdin) && isTerminal(s.Stdout)
}

// Startup warnings when the home directory is unknown and no absolute XDG
// directory replaces it.
const (
	noHomeHistory  = "Cannot determine the home directory; conversation history is disabled."
	noHomeSettings = "Cannot determine the home directory; settings are not loaded or saved."
)

// runInteractive starts the fullscreen UI or the plain chat: it restores
// the conversation for the requested selection, checks the OpenAI key,
// picks the model, applies the saved settings under the CLI flags, and
// runs the chat. Problems with history or settings files are warnings; a
// missing key, a failed model lookup or an unsupported CLI setting end
// the run before the chat starts, so the terminal is never touched then.
func runInteractive(ctx context.Context, opts options, s Streams) int {
	startCtx, stop := cancelOnInterrupt(ctx, s.Interrupts)
	defer stop()

	provider, model := opts.provider, opts.model
	if model != "" && provider == "" {
		provider = providers[0] // --model alone means the default provider
	}

	home, _ := os.UserHomeDir()
	paths, pathErr := storage.ResolvePaths(os.Getenv, home)
	var warnings []string
	var history *storage.HistoryStore
	switch {
	case opts.noHistory:
		// No history reads, writes or permission changes at all.
	case pathErr != nil:
		warnings = append(warnings, noHomeHistory)
	default:
		history = storage.NewHistoryStore(paths.Conversations)
	}
	sess, sessWarnings := session.Open(session.Options{Store: history, Provider: provider, Model: model})
	warnings = append(warnings, sessWarnings...)

	sel := sess.Selection()
	if sel.Provider == "" {
		sel.Provider = providers[0]
	}
	if sel.Provider == "openai" && os.Getenv(openai.APIKeyEnvVar) == "" {
		return reportError(startCtx, s.Stderr, openai.MissingAPIKeyError())
	}
	p, err := newProvider(sel.Provider)
	if err != nil {
		return usageError(s.Stderr, err.Error())
	}
	if sel.Model == "" {
		if sel.Model, err = p.DefaultModel(startCtx); err != nil {
			return reportError(startCtx, s.Stderr, err)
		}
	}
	sess.SetDefaults(sel.Provider, sel.Model)

	saved := storage.DefaultSettings()
	var settingsStore *storage.SettingsStore
	if pathErr != nil {
		warnings = append(warnings, noHomeSettings)
	} else {
		// Only the fullscreen UI's F4–F6 write it, after a confirmed change.
		var settingsWarnings []string
		settingsStore = storage.NewSettingsStore(paths.Settings)
		saved, settingsWarnings = settingsStore.Load()
		warnings = append(warnings, settingsWarnings...)
	}
	overrides := chat.Overrides{
		Reasoning:       opts.reasoning.value,
		MaxOutputTokens: opts.maxOutputTokens,
		AnswerStyle:     opts.style.value,
	}
	settings, notices, err := chat.ResolveSettings(startCtx, p, sel.Model, saved, overrides)
	if err != nil {
		return reportError(startCtx, s.Stderr, err)
	}
	warnings = append(warnings, notices...)
	useTUI := fullscreen(opts, s)
	var support llm.ReasoningSupport
	if useTUI {
		// For the settings panel; cached, so requests do not ask again.
		support, _ = p.ReasoningSupport(startCtx, sel.Model)
	}
	// The chat handles Ctrl+C itself from here on; one pressed during the
	// startup checks ends the run.
	if stop() {
		return reportError(startCtx, s.Stderr, context.Canceled)
	}

	engine := &chat.Engine{Provider: p, Model: sel.Model, Settings: settings}
	if useTUI {
		return runTUI(ctx, opts, s, tui.Config{
			Title:       "NeuroBräu Go CLI " + Version,
			Session:     sess,
			Engine:      engine,
			Reasoning:   settings.EffectiveReasoning(support),
			Preferences: tui.Preferences{Store: settingsStore, Saved: saved, Overrides: overrides},
			NewProvider: newProvider,
			Warnings:    warnings,
		})
	}
	cfg := chat.Config{
		Title:      "NeuroBräu Go CLI " + Version,
		Session:    sess,
		Reply:      engine.Reply,
		Warnings:   warnings,
		ShowUsage:  opts.usage,
		Stdin:      s.Stdin,
		Stdout:     s.Stdout,
		Stderr:     s.Stderr,
		Interrupts: s.Interrupts,
	}
	if opts.stream {
		cfg.Stream = engine.Stream
	}
	return chat.Run(ctx, cfg)
}

// runTUI initializes the terminal and runs the fullscreen UI, which
// restores the terminal on every exit path.
func runTUI(ctx context.Context, opts options, s Streams, cfg tui.Config) int {
	// Watched before the terminal changes, so a signal never leaves it in
	// raw mode; one that arrives after the UI ended still sets the code.
	terminations := make(chan os.Signal, 1)
	if s.WatchTerminations != nil {
		defer s.WatchTerminations(terminations)()
		cfg.Terminations = terminations
	}
	pending := func(code int) int {
		select {
		case sig := <-terminations:
			return tui.SignalExitCode(sig)
		default:
			return code
		}
	}
	screen, err := newScreen()
	if err == nil {
		err = screen.Init()
	}
	if err != nil {
		fmt.Fprintf(s.Stderr, "%s: error: cannot start the fullscreen UI: %v (use --plain for the line-oriented chat)\n", ProgName, err)
		return pending(chat.ExitError)
	}
	cfg.Screen = screen
	cfg.Stream = opts.stream
	cfg.TranscriptOnExit = opts.transcript
	cfg.Stdout, cfg.Stderr, cfg.Interrupts = s.Stdout, s.Stderr, s.Interrupts
	return pending(tui.Run(ctx, cfg))
}
