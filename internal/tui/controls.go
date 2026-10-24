package tui

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/gdamore/tcell/v2"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// providerNames are the providers of the F3 menu, in menu order.
var providerNames = []string{storage.ProviderOpenAI, storage.ProviderOllama}

// tokenPresets are the F5 presets (docs/tui-spec.md, section 4c).
var tokenPresets = []int{256, 512, 1024, 2048, 4096}

var styleNotes = map[string]string{
	llm.StyleBrief:    "short, concise answers",
	llm.StyleNormal:   "standard answers",
	llm.StyleDetailed: "thorough, with examples",
}

// functionKey opens the F2–F6 menus. Settings never change while a
// request or check runs, or before a canceled one has closed.
func (a *app) functionKey(key tcell.Key) {
	switch {
	case a.active != nil:
		a.flashWarning("Settings cannot change while a response is generated")
		return
	case a.op != nil:
		a.flashWarning("Wait for the current check or press Ctrl+C")
		return
	case !a.idle():
		return
	}
	a.transcript.scroll = 0
	switch key {
	case tcell.KeyF2:
		a.loadModels()
	case tcell.KeyF3:
		a.openProviderMenu()
	case tcell.KeyF4:
		a.checkReasoning()
	case tcell.KeyF5:
		a.checkLimits()
	case tcell.KeyF6:
		a.openStyleMenu()
	}
}

// menuKey handles a key while a menu is open.
func (a *app) menuKey(ev *tcell.EventKey) {
	if ev.Key() == tcell.KeyCtrlC {
		a.exit(ExitInterrupted)
		return
	}
	visible := 1
	if l, ok := a.layout(); ok {
		visible = menuVisible(l)
	}
	m := a.menu
	switch m.key(ev.Key(), visible) {
	case menuCancel:
		a.menu = nil
	case menuSelect:
		a.menu = nil
		value := m.selected().value
		switch m.kind {
		case menuProvider:
			a.selectProvider(value)
		case menuModel:
			a.selectModel(value)
		case menuReasoning:
			a.selectReasoning(value)
		case menuTokens:
			a.selectTokens(value)
		case menuStyle:
			a.selectStyle(value)
		}
	}
}

func (a *app) selection() storage.Selection { return a.Session.Selection() }

// provider returns the provider with that name, created once (no request).
func (a *app) provider(name string) (llm.StreamingProvider, error) {
	if p, ok := a.providers[name]; ok {
		return p, nil
	}
	p, err := a.NewProvider(name)
	if err != nil {
		return nil, err
	}
	a.providers[name] = p
	return p, nil
}

// --- F3 provider and F2 model ---

// openProviderMenu lists the providers without any request.
func (a *app) openProviderMenu() {
	current := a.selection().Provider
	items := make([]menuItem, len(providerNames))
	for i, name := range providerNames {
		items[i] = menuItem{label: providerLabel(name), value: name, current: name == current}
	}
	a.menu = newMenu(menuProvider, "SELECT PROVIDER", items)
}

// selectProvider checks that another provider can be used, then switches
// to its default model. The current one stays if it cannot.
func (a *app) selectProvider(name string) {
	if name == a.selection().Provider {
		return
	}
	label := providerLabel(name)
	p, err := a.provider(name)
	if err != nil {
		a.addEntry(entryError, fmt.Sprintf("%s is not available: %v", label, err))
		return
	}
	prefs := a.Preferences // a copy for the goroutine
	a.startOp("Checking "+label, func(ctx context.Context) func() {
		unavailable := func(err error) func() {
			return func() { a.addEntry(entryError, fmt.Sprintf("%s is not available: %v", label, err)) }
		}
		if err := checkAvailability(ctx, p); err != nil {
			return unavailable(err)
		}
		model, err := p.DefaultModel(ctx)
		if err != nil {
			return unavailable(err)
		}
		r := resolveFor(ctx, p, model, prefs)
		return func() { a.switchTo(name, p, model, r) }
	})
}

func checkAvailability(ctx context.Context, p llm.Provider) error {
	if c, ok := p.(llm.AvailabilityChecker); ok {
		return c.CheckAvailability(ctx)
	}
	_, err := p.ListModels(ctx)
	return err
}

// loadModels discovers the current provider's models (cached by the
// provider), then opens the model menu.
func (a *app) loadModels() {
	sel := a.selection()
	label := providerLabel(sel.Provider)
	p := a.Engine.Provider
	a.startOp("Loading "+label+" models", func(ctx context.Context) func() {
		models, err := p.ListModels(ctx)
		return func() {
			switch {
			case err != nil:
				a.showError(err.Error())
			case len(models) == 0:
				a.addEntry(entryError, fmt.Sprintf("No selectable %s models were found.", label))
			default:
				items := make([]menuItem, len(models))
				for i, m := range models {
					items[i] = menuItem{label: m.ID, value: m.ID, note: m.Details, current: m.ID == sel.Model}
				}
				a.menu = newMenu(menuModel, "SELECT "+strings.ToUpper(label)+" MODEL", items)
			}
		}
	})
}

func (a *app) selectModel(model string) {
	sel := a.selection()
	if model == sel.Model {
		return
	}
	p, prefs := a.Engine.Provider, a.Preferences
	a.startOp("Checking model settings", func(ctx context.Context) func() {
		r := resolveFor(ctx, p, model, prefs)
		return func() { a.switchTo(sel.Provider, p, model, r) }
	})
}

// resolved are a model's effective settings after a switch.
type resolved struct {
	settings  chat.Settings
	reasoning string // for display
	notices   []string
	err       error
}

// resolveFor works out a model's settings: its own saved choices under
// this run's remaining command-line overrides; values the model does not
// support become notices. It may make cached requests.
func resolveFor(ctx context.Context, p llm.Provider, model string, prefs Preferences) resolved {
	settings, notices, err := chat.ResolveSwitchedSettings(ctx, p, model, prefs.Saved, prefs.Overrides)
	if err != nil {
		return resolved{err: err}
	}
	support, err := p.ReasoningSupport(ctx, model)
	if err != nil && ctx.Err() != nil {
		return resolved{err: err}
	}
	return resolved{settings: settings, reasoning: settings.EffectiveReasoning(support), notices: notices}
}

// switchTo continues with another provider/model: its own conversation
// comes back (from this run, else the latest saved one), or a new one
// starts. Messages never move between selections.
func (a *app) switchTo(provider string, p llm.Provider, model string, r resolved) {
	if r.err != nil {
		a.showError(r.err.Error())
		return
	}
	hadMessages := len(a.Session.Messages()) > 0
	saveErr := a.Session.Switch(provider, model)
	a.Engine.Provider, a.Engine.Model, a.Engine.Settings = p, model, r.settings
	a.reasoning = r.reasoning
	a.confirming = false

	a.transcript.entries, a.transcript.scroll = nil, 0
	nowUsing := fmt.Sprintf("Now using %s / %s.", providerLabel(provider), model)
	if a.Session.Restored() {
		messages := a.Session.Messages()
		a.showMessages(messages)
		a.addEntry(entryInfo, fmt.Sprintf("%s Restored its conversation (%d messages).", nowUsing, len(messages)))
	} else {
		notice := nowUsing + " Started a new conversation"
		switch {
		case hadMessages && a.Session.Persistent():
			notice += "; the previous one stays saved."
		case hadMessages:
			notice += "; the previous messages are not sent to the new model."
		default:
			notice += "."
		}
		a.addEntry(entryInfo, notice)
	}
	for _, n := range r.notices {
		a.addEntry(entryInfo, n)
	}
	a.status = ready()
	if saveErr != nil {
		a.addEntry(entryError, "Warning: "+saveErr.Error())
		a.status = warning("Conversation not saved")
	}
}

// --- F4 reasoning, F5 max output tokens, F6 answer style ---

func (a *app) checkReasoning() {
	p, model := a.Engine.Provider, a.Engine.Model
	a.startOp("Checking reasoning support", func(ctx context.Context) func() {
		support, err := p.ReasoningSupport(ctx, model)
		return func() {
			if err != nil {
				a.showError(err.Error())
				return
			}
			a.openReasoningMenu(model, support)
		}
	})
}

func (a *app) openReasoningMenu(model string, support llm.ReasoningSupport) {
	var items []menuItem
	switch {
	case !support.Known:
		items = []menuItem{{label: "Model default (support unknown)", current: true}}
		a.addEntry(entryInfo, fmt.Sprintf("Reasoning support for %s is not confirmed, so only the model's default "+
			"is offered. --reasoning-effort sends a value explicitly.", model))
	case len(support.Values) == 0:
		a.addEntry(entryInfo, model+" does not support reasoning.")
		return
	default:
		// "Model default" sends no parameter, so the model decides; it is not
		// NeuroBräu's own default ("none" or "off" where supported).
		if !support.NoModelDefault {
			note := "no effort sent"
			if support.ModelDefault != "" {
				note += ": " + support.ModelDefault
			}
			items = append(items, menuItem{label: "Model default", note: note})
		}
		for _, v := range support.Values {
			items = append(items, menuItem{label: llm.ReasoningLabel(v), value: v})
		}
		for i := range items {
			items[i].current = items[i].value == a.reasoning
		}
	}
	a.menu = newMenu(menuReasoning, "REASONING", items)
}

// selectReasoning applies a reasoning value ("" sends no parameter) to the
// next requests and saves it for the model ("default" for "").
func (a *app) selectReasoning(value string) {
	sel := a.selection()
	prefs := a.Preferences.Saved.ForModel(sel.Provider, sel.Model)
	saved := value
	if value == "" {
		saved = storage.ReasoningDefault
	}
	unchanged := value == a.reasoning && a.Preferences.Overrides.Reasoning == "" &&
		(prefs.Reasoning == saved || prefs.Reasoning == "" && a.Engine.Settings.Reasoning == llm.Reasoning{})
	if unchanged {
		return
	}
	a.Engine.Settings.Reasoning = llm.Reasoning{Effort: value, Explicit: true}
	a.reasoning = value
	a.Preferences.Overrides.Reasoning = "" // an interactive choice replaces the flag
	prefs.Reasoning = saved
	a.Preferences.Saved = a.Preferences.Saved.WithModel(sel.Provider, sel.Model, prefs)
	chosen := llm.ReasoningLabel(value)
	if value == "" {
		chosen += " (no effort parameter is sent)"
	}
	a.addEntry(entryInfo, fmt.Sprintf("Reasoning for %s: %s. Applies from the next message.", sel.Model, chosen))
	a.savePreferences()
}

func (a *app) checkLimits() {
	p, model := a.Engine.Provider, a.Engine.Model
	a.startOp("Checking model limits", func(ctx context.Context) func() {
		limit, err := p.MaxOutputLimit(ctx, model)
		return func() {
			if err != nil {
				a.showError(err.Error())
				return
			}
			a.openTokensMenu(limit)
		}
	})
}

// openTokensMenu offers Default, the presets within the model's known
// limit (0: unknown) and a custom value.
func (a *app) openTokensMenu(limit int) {
	a.tokenLimit = limit
	current := a.Engine.Settings.MaxOutputTokens
	items := []menuItem{{label: "Default", note: "model default", current: current == 0}}
	var presets []int
	for _, n := range tokenPresets {
		if limit == 0 || n <= limit {
			presets = append(presets, n)
			items = append(items, menuItem{label: strconv.Itoa(n), value: strconv.Itoa(n), current: n == current})
		}
	}
	custom := menuItem{label: "Custom...", value: "custom", note: "any positive number"}
	if limit > 0 {
		custom.note = fmt.Sprintf("up to %d", limit)
	}
	if current > 0 && !slices.Contains(presets, current) {
		custom.label, custom.current = fmt.Sprintf("Custom (%d)", current), true
	}
	a.menu = newMenu(menuTokens, "MAX OUTPUT TOKENS", append(items, custom))
}

func (a *app) selectTokens(value string) {
	if value == "custom" {
		a.startTokenPrompt()
		return
	}
	n, _ := strconv.Atoi(value) // "" (Default) is 0
	a.applyTokens(n)
}

// applyTokens applies a max output tokens value (0: the model's default)
// and saves it for the model (0 as JSON null).
func (a *app) applyTokens(n int) {
	sel := a.selection()
	prefs := a.Preferences.Saved.ForModel(sel.Provider, sel.Model)
	unchanged := n == a.Engine.Settings.MaxOutputTokens && a.Preferences.Overrides.MaxOutputTokens == 0 &&
		(prefs.MaxOutputTokensSet && prefs.MaxOutputTokens == n || !prefs.MaxOutputTokensSet && n == 0)
	if unchanged {
		return
	}
	a.Engine.Settings.MaxOutputTokens = n
	a.Preferences.Overrides.MaxOutputTokens = 0
	prefs.MaxOutputTokensSet, prefs.MaxOutputTokens = true, n
	a.Preferences.Saved = a.Preferences.Saved.WithModel(sel.Provider, sel.Model, prefs)
	value := "model default"
	if n > 0 {
		value = strconv.Itoa(n)
	}
	a.addEntry(entryInfo, fmt.Sprintf("Max output tokens for %s: %s. Applies from the next message.", sel.Model, value))
	a.savePreferences()
}

// startTokenPrompt turns the input line into the custom token prompt; the
// typed text is set aside and restored afterwards.
func (a *app) startTokenPrompt() {
	a.draft, a.tokenPrompt = a.input, true
	a.input = inputLine{}
	limit := ""
	if a.tokenLimit > 0 {
		limit = fmt.Sprintf(" (1-%d)", a.tokenLimit)
	}
	a.addEntry(entryInfo, fmt.Sprintf("Type the max output tokens%s and press Enter; Esc cancels.", limit))
}

func (a *app) endTokenPrompt() {
	a.input, a.tokenPrompt = a.draft, false
	a.draft = inputLine{}
}

// submitTokens checks a typed token limit; after an error the prompt
// stays open.
func (a *app) submitTokens(text string) {
	retry := " Try again or press Esc."
	if text == "" || strings.ContainsFunc(text, func(r rune) bool { return r < '0' || r > '9' }) {
		a.addEntry(entryError, fmt.Sprintf("'%s' is not a positive whole number.%s", text, retry))
		return
	}
	model := a.Engine.Model
	n, err := strconv.Atoi(text)
	if err != nil { // too large for an int
		if a.tokenLimit > 0 {
			a.addEntry(entryError, fmt.Sprintf("%s exceeds the limit of %d output tokens for %s.%s", text, a.tokenLimit, model, retry))
		} else {
			a.addEntry(entryError, fmt.Sprintf("'%s' is too large.%s", text, retry))
		}
		return
	}
	if err := llm.CheckMaxOutputTokens(model, n, a.tokenLimit); err != nil {
		a.addEntry(entryError, err.Error()+retry)
		return
	}
	a.endTokenPrompt()
	a.applyTokens(n)
}

func (a *app) openStyleMenu() {
	items := make([]menuItem, len(llm.AnswerStyles))
	for i, s := range llm.AnswerStyles {
		items[i] = menuItem{label: capitalize(s), value: s, note: styleNotes[s], current: s == a.Engine.Settings.AnswerStyle}
	}
	a.menu = newMenu(menuStyle, "ANSWER STYLE", items)
}

// selectStyle applies the global answer style and saves it.
func (a *app) selectStyle(style string) {
	if style == a.Engine.Settings.AnswerStyle && style == a.Preferences.Saved.AnswerStyle {
		return
	}
	a.Engine.Settings.AnswerStyle = style
	a.Preferences.Overrides.AnswerStyle = ""
	a.Preferences.Saved.AnswerStyle = style
	a.addEntry(entryInfo, fmt.Sprintf("Answer style: %s. Applies from the next message.", capitalize(style)))
	a.savePreferences()
}

// savePreferences writes settings.json. A failure leaves the choice in
// effect for this run and is shown as a warning.
func (a *app) savePreferences() {
	store := a.Preferences.Store
	if store == nil {
		a.addEntry(entryError, "Warning: Settings not saved: the settings location is unknown.")
		return
	}
	if err := store.Save(a.Preferences.Saved); err != nil {
		a.addEntry(entryError, "Warning: "+err.Error())
		a.status = warning("Settings not saved")
	}
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
