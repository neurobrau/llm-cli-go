package cli

import (
	"context"
	"os"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/openai"
	"github.com/neurobrau/llm-cli-go/internal/session"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// Startup warnings when the home directory is unknown and no absolute XDG
// directory replaces it.
const (
	noHomeHistory  = "Cannot determine the home directory; conversation history is disabled."
	noHomeSettings = "Cannot determine the home directory; settings are not loaded or saved."
)

// runInteractive starts the plain chat: it restores the conversation for
// the requested selection, checks the OpenAI key, picks the model, applies
// the saved settings under the CLI flags, and runs the chat loop. Problems
// with history or settings files are warnings; a missing key, a failed
// model lookup or an unsupported CLI setting end the run before the chat.
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
	if pathErr != nil {
		warnings = append(warnings, noHomeSettings)
	} else {
		// Read only: a chat without settings controls never writes them.
		var settingsWarnings []string
		saved, settingsWarnings = storage.NewSettingsStore(paths.Settings).Load()
		warnings = append(warnings, settingsWarnings...)
	}
	settings, notices, err := chat.ResolveSettings(startCtx, p, sel.Model, saved, chat.Overrides{
		Reasoning:       opts.reasoning.value,
		MaxOutputTokens: opts.maxOutputTokens,
		AnswerStyle:     opts.style.value,
	})
	if err != nil {
		return reportError(startCtx, s.Stderr, err)
	}
	warnings = append(warnings, notices...)
	// The chat handles Ctrl+C itself from here on; one pressed during the
	// startup checks ends the run.
	if stop() {
		return reportError(startCtx, s.Stderr, context.Canceled)
	}

	engine := &chat.Engine{Provider: p, Model: sel.Model, Settings: settings}
	return chat.Run(ctx, chat.Config{
		Title:      "NeuroBräu Go CLI " + Version,
		Session:    sess,
		Reply:      engine.Reply,
		Warnings:   warnings,
		ShowUsage:  opts.usage,
		Stdin:      s.Stdin,
		Stdout:     s.Stdout,
		Stderr:     s.Stderr,
		Interrupts: s.Interrupts,
	})
}
