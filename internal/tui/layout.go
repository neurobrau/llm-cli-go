// Package tui is the DOS-style fullscreen chat (docs/tui-spec.md,
// sections 1–10 and 14): layout, wrapping, input editing and the event
// loop that draws on a tcell.Screen. Requests run in worker goroutines;
// only the event loop draws or changes what the screen shows.
// Conversations, providers and settings stay in their own packages.
package tui

import "strings"

// Geometry (docs/tui-spec.md, section 2).
const (
	MinCols = 40
	MinRows = 12

	chromeRows    = 7  // top border, 2 separators, status, input, bottom border, help line
	panelWidth    = 22 // settings panel: separator, space, 18 content columns, space, border
	panelMinCols  = 64 // narrower terminals hide the panel
	panelContent  = panelWidth - 4
	conversationY = 1 // first conversation row
)

// Layout is the screen geometry for one terminal size.
type Layout struct {
	Rows, Cols int
}

// NewLayout returns the layout for a terminal size; false if it is
// smaller than MinCols x MinRows.
func NewLayout(cols, rows int) (Layout, bool) {
	if cols < MinCols || rows < MinRows {
		return Layout{}, false
	}
	return Layout{Rows: rows, Cols: cols}, true
}

// PanelX is the column of the separator left of the LLM SETTINGS panel;
// false if the panel is hidden.
func (l Layout) PanelX() (int, bool) {
	if l.Cols < panelMinCols {
		return 0, false
	}
	return l.Cols - panelWidth, true
}

// ConversationLimit is the exclusive right edge of conversation text.
func (l Layout) ConversationLimit() int {
	if x, ok := l.PanelX(); ok {
		return x - 1
	}
	return l.Cols - 2
}

// ConversationWidth is the width of conversation text (from column 2).
func (l Layout) ConversationWidth() int { return l.ConversationLimit() - 2 }

// ConversationHeight is the number of conversation rows.
func (l Layout) ConversationHeight() int { return l.Rows - chromeRows }

// Rows of the regions below the conversation.
func (l Layout) StatusSeparatorRow() int { return conversationY + l.ConversationHeight() }
func (l Layout) StatusRow() int          { return l.StatusSeparatorRow() + 1 }
func (l Layout) InputSeparatorRow() int  { return l.StatusRow() + 1 }
func (l Layout) InputRow() int           { return l.InputSeparatorRow() + 1 }
func (l Layout) BottomRow() int          { return l.InputRow() + 1 }
func (l Layout) HelpRow() int            { return l.Rows - 1 }

// InnerWidth is the width of the status and input lines (from column 2).
func (l Layout) InnerWidth() int { return l.Cols - 4 }

// HorizontalRule is a `+-[ title ]----+` border of exactly width cells
// (no title: `+----+`).
func HorizontalRule(width int, title string) string {
	if width < 2 {
		return strings.Repeat("+", max(width, 0))
	}
	inner := []rune("-")
	if title != "" {
		inner = append(inner, []rune("[ "+title+" ]")...)
	}
	for len(inner) < width-2 {
		inner = append(inner, '-')
	}
	return "+" + string(inner[:width-2]) + "+"
}

// fit clips text to width cells, marking a cut with `~`.
func fit(text string, width int) string {
	r := []rune(text)
	if len(r) <= width {
		return text
	}
	return string(r[:max(0, width-1)]) + "~"
}
