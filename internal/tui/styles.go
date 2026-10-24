package tui

import "github.com/gdamore/tcell/v2"

// role is a visual role of docs/tui-spec.md, section 3.
type role int

const (
	roleText role = iota
	roleBorder
	roleUser
	roleAI
	roleInfo
	roleError
	roleWaiting
	roleSelected // highlighted menu item
)

// styles maps each role to a tcell style; everything is drawn on black.
type styles [roleSelected + 1]tcell.Style

// newStyles uses the 8 basic colors (as curses does) on a black
// background, or bold/reverse on a monochrome terminal.
func newStyles(colors int) styles {
	base := tcell.StyleDefault.Background(tcell.ColorBlack).Foreground(tcell.ColorSilver)
	var s styles
	if colors < 8 {
		base = tcell.StyleDefault
		for i := range s {
			s[i] = base
		}
		for _, r := range []role{roleBorder, roleUser, roleAI, roleError} {
			s[r] = base.Bold(true)
		}
		s[roleWaiting] = base.Reverse(true)
		s[roleSelected] = base.Reverse(true)
		return s
	}
	s[roleText] = base
	s[roleBorder] = base.Foreground(tcell.ColorGreen).Bold(true)
	s[roleUser] = base.Foreground(tcell.ColorNavy).Bold(true)
	s[roleAI] = base.Foreground(tcell.ColorGreen).Bold(true)
	s[roleInfo] = base.Foreground(tcell.ColorGreen)
	s[roleError] = base.Foreground(tcell.ColorMaroon).Bold(true)
	s[roleWaiting] = base.Foreground(tcell.ColorOlive).Bold(true)
	s[roleSelected] = base.Reverse(true)
	return s
}
