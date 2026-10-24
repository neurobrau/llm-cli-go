package tui

import (
	"fmt"
	"strings"

	"github.com/gdamore/tcell/v2"
)

// menuKind says what selecting an item of a menu means.
type menuKind int

const (
	menuProvider menuKind = iota
	menuModel
	menuReasoning
	menuTokens
	menuStyle
)

const menuMinWidth = 30

// menuItem is one choice; value is what selecting it means to the menu's
// kind.
type menuItem struct {
	label   string
	value   string
	note    string // shown right-aligned when it fits
	current bool   // marked with `*`
}

// menu is a selection menu drawn over the conversation (docs/tui-spec.md,
// section 4b): a list with a highlighted item and a scroll offset. It
// holds no drawing state; render draws it from scratch.
type menu struct {
	kind  menuKind
	title string
	items []menuItem
	index int // highlighted item
	top   int // first visible item
}

// newMenu opens with the current item highlighted.
func newMenu(kind menuKind, title string, items []menuItem) *menu {
	index := 0
	for i, item := range items {
		if item.current {
			index = i
			break
		}
	}
	return &menu{kind: kind, title: title, items: items, index: index, top: index}
}

func (m *menu) selected() menuItem { return m.items[m.index] }

// moveTo highlights index (clamped) and scrolls it into view.
func (m *menu) moveTo(index, visible int) {
	m.index = max(0, min(len(m.items)-1, index))
	m.scrollIntoView(visible)
}

func (m *menu) scrollIntoView(visible int) {
	visible = max(1, visible)
	if m.index < m.top {
		m.top = m.index
	} else if m.index >= m.top+visible {
		m.top = m.index - visible + 1
	}
	m.top = max(0, min(m.top, max(0, len(m.items)-visible)))
}

type menuAction int

const (
	menuNone menuAction = iota
	menuSelect
	menuCancel
)

// key applies a key press; visible is how many items fit on screen. Keys
// other than the menu's are ignored, so nothing reaches the input line.
func (m *menu) key(k tcell.Key, visible int) menuAction {
	page := max(1, visible-1)
	switch k {
	case tcell.KeyEnter, tcell.KeyCtrlJ:
		return menuSelect
	case tcell.KeyEsc, tcell.KeyCtrlD:
		return menuCancel
	case tcell.KeyUp:
		m.moveTo(m.index-1, visible)
	case tcell.KeyDown:
		m.moveTo(m.index+1, visible)
	case tcell.KeyPgUp:
		m.moveTo(m.index-page, visible)
	case tcell.KeyPgDn:
		m.moveTo(m.index+page, visible)
	case tcell.KeyHome:
		m.moveTo(0, visible)
	case tcell.KeyEnd:
		m.moveTo(len(m.items)-1, visible)
	}
	return menuNone
}

// menuLine is `* label    note` padded to width; width 0 gives the
// natural length. The note is left out when it does not fit.
func menuLine(item menuItem, width int) string {
	marker := " "
	if item.current {
		marker = "*"
	}
	left := marker + " " + item.label
	n := len([]rune(left))
	if width <= 0 {
		if item.note != "" {
			return left + "  " + item.note
		}
		return left
	}
	note := ""
	if item.note != "" && n+len([]rune(item.note))+2 <= width {
		note = "  " + item.note
	}
	rest := width - len([]rune(note))
	left = fit(left, rest)
	return left + strings.Repeat(" ", rest-len([]rune(left))) + note
}

// menuVisible is how many items fit: the conversation height minus the
// box's borders.
func menuVisible(l Layout) int { return max(1, l.ConversationHeight()-2) }

// renderMenu draws the open menu as a bordered box centered over the
// conversation area if it fits there, else over the whole width.
func (a *app) renderMenu(l Layout) {
	m := a.menu
	if m == nil {
		return
	}
	visible := menuVisible(l)
	m.scrollIntoView(visible)
	shown := m.items[m.top:min(len(m.items), m.top+visible)]
	footer := fmt.Sprintf("%d/%d Enter=OK Esc=cancel", m.index+1, len(m.items))
	longest := 0
	for _, item := range m.items {
		longest = max(longest, len([]rune(menuLine(item, 0))))
	}
	// A rule with a title is the title + 7 wide (`+-[ title ]+`); the spec's
	// + 6 would cut the closing bracket.
	width := min(l.Cols-2, max(menuMinWidth, longest+4, len([]rune(m.title))+7, len([]rune(footer))+7))
	height := len(shown) + 2
	area := l.Cols
	if panelX, ok := l.PanelX(); ok && width <= panelX-2 {
		area = panelX
	}
	x0 := (area - width) / 2
	y0 := conversationY + max(0, (l.ConversationHeight()-height)/2)
	right := x0 + width
	a.put(x0, y0, HorizontalRule(width, m.title), roleBorder, right)
	for i, item := range shown {
		y := y0 + 1 + i
		role := roleText
		if m.top+i == m.index {
			role = roleSelected
		}
		a.put(x0, y, "|", roleBorder, right)
		a.put(x0+1, y, " "+menuLine(item, width-4)+" ", role, right-1)
		a.put(right-1, y, "|", roleBorder, right)
	}
	a.put(x0, y0+height-1, HorizontalRule(width, footer), roleBorder, right)
}
