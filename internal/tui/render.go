package tui

import (
	"fmt"
	"unicode"
)

// render redraws the whole screen from the state (docs/tui-spec.md,
// sections 2–6); tcell sends only the changed cells to the terminal.
func (a *app) render() {
	s := a.Screen
	s.SetStyle(a.styles[roleText])
	s.Clear()
	cols, rows := s.Size()
	l, ok := NewLayout(cols, rows)
	if !ok {
		a.renderTooSmall(cols, rows)
	} else {
		a.transcript.scroll = min(a.transcript.scroll, a.maxScroll())
		a.renderFrame(l)
		a.renderConversation(l)
		a.renderPanel(l)
		a.renderStatus(l)
		a.renderInput(l)
		a.renderMenu(l)
	}
	s.Show()
}

func (a *app) renderTooSmall(cols, rows int) {
	a.Screen.HideCursor()
	lines := []string{
		"Terminal too small.",
		fmt.Sprintf("Need %dx%d, have %dx%d.", MinCols, MinRows, cols, rows),
		"Resize or press Ctrl+C.",
	}
	for y, text := range lines[:min(len(lines), rows)] {
		// One column short, so the bottom-right cell is never written.
		a.put(0, y, text, roleError, cols-1)
	}
}

func (a *app) renderFrame(l Layout) {
	title := a.Title
	if a.transcript.scroll > 0 {
		title += fmt.Sprintf(" - scrolled up %d (PgDn)", a.transcript.scroll)
	}
	w := l.Cols
	panelX, panel := l.PanelX()
	top := HorizontalRule(w, title)
	if panel {
		top = HorizontalRule(panelX+1, title) + string([]rune(HorizontalRule(w-panelX, "LLM SETTINGS"))[1:])
	}
	a.put(0, 0, top, roleBorder, w)
	for y := conversationY; y < l.StatusSeparatorRow(); y++ {
		a.put(0, y, "|", roleBorder, w)
		if panel {
			a.put(panelX, y, "|", roleBorder, w)
		}
		a.put(w-1, y, "|", roleBorder, w)
	}
	a.put(0, l.StatusSeparatorRow(), HorizontalRule(w, "Status"), roleBorder, w)
	if panel {
		a.put(panelX, l.StatusSeparatorRow(), "+", roleBorder, w)
	}
	for _, y := range []int{l.StatusRow(), l.InputRow()} {
		a.put(0, y, "|", roleBorder, w)
		a.put(w-1, y, "|", roleBorder, w)
	}
	a.put(0, l.InputSeparatorRow(), HorizontalRule(w, "Input"), roleBorder, w)
	a.put(0, l.BottomRow(), HorizontalRule(w, ""), roleBorder, w)
	// The last cell of the screen is never written (terminals may scroll).
	a.put(0, l.HelpRow(), HelpLine, roleBorder, w-1)
}

func (a *app) renderConversation(l Layout) {
	lines := a.transcript.lines(l.ConversationWidth())
	start, end := visibleWindow(len(lines), l.ConversationHeight(), a.transcript.scroll)
	for i, ln := range lines[start:end] {
		x := 2
		for _, seg := range ln {
			x = a.put(x, conversationY+i, seg.text, seg.role, l.ConversationLimit())
		}
	}
}

func (a *app) renderPanel(l Layout) {
	panelX, ok := l.PanelX()
	if !ok {
		return
	}
	rows := settingsPanel(a.Session.Selection(), a.reasoning, a.Engine.Settings, a.Session.Requests())
	for i, row := range rows[:min(len(rows), l.ConversationHeight())] {
		a.put(panelX+2, conversationY+i, row.text, row.role, l.Cols-2)
	}
}

// currentStatus is what the runtime bar shows: a flashed warning, then
// live activity, then the stored status.
func (a *app) currentStatus() status {
	now := a.now()
	if a.flash != nil {
		if now.Before(a.flashUntil) {
			return *a.flash
		}
		a.flash = nil
	}
	if a.op != nil {
		return status{stateConnecting, []string{a.op.label}}
	}
	if req := a.active; req != nil {
		if !req.streamed {
			return status{stateGenerating, []string{"Waiting for the complete response", cancelHint}}
		}
		elapsed := now.Sub(req.started)
		switch {
		case !req.firstText.IsZero():
			return activeStatus(stateGenerating, elapsed, "")
		case req.thinking:
			return activeStatus(stateGenerating, elapsed, "Thinking")
		default:
			return activeStatus(stateConnecting, elapsed, "")
		}
	}
	if a.confirming {
		return warning("Type yes to delete saved conversations; anything else cancels")
	}
	if a.tokenPrompt {
		return ready("Type a token limit and press Enter", "Esc cancels")
	}
	return a.status
}

func (a *app) renderStatus(l Layout) {
	st := a.currentStatus()
	x := a.put(2, l.StatusRow(), st.state, st.role(), l.Cols-2)
	for _, part := range st.parts {
		x = a.put(x, l.StatusRow(), statusSeparator+part, roleText, l.Cols-2)
	}
}

func (a *app) renderInput(l Layout) {
	label := userLabel
	if a.tokenPrompt {
		label = tokensLabel
	}
	x := a.put(2, l.InputRow(), label, roleUser, l.Cols-2)
	visible, cursor := a.input.view(l.InnerWidth() - len(label))
	a.put(x, l.InputRow(), visible, roleText, l.Cols-2)
	a.Screen.ShowCursor(x+cursor, l.InputRow())
}

// put draws text from column x, one cell per rune, never at or past
// column limit, and returns the column after it. Control characters
// (e.g. from a model's answer) are drawn as `?`, so they cannot reach the
// terminal.
func (a *app) put(x, y int, text string, role role, limit int) int {
	style := a.styles[role]
	for _, r := range text {
		if x >= limit {
			break
		}
		if unicode.IsControl(r) {
			r = '?'
		}
		a.Screen.SetContent(x, y, r, nil, style)
		x++
	}
	return x
}
