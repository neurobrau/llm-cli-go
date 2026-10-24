package tui

import "strings"

// Entry kinds of the conversation panel (docs/tui-spec.md, section 4).
type entryKind int

const (
	entryUser entryKind = iota
	entryAI
	entryInfo
	entryError
)

// Labels of user and assistant entries and of the input line.
const (
	userLabel = "YOU > "
	aiLabel   = "AI > "
	// tokensLabel replaces userLabel during the custom token prompt.
	tokensLabel = "TOKENS > "
)

// entry is one item in the conversation panel. Its wrapped lines are
// cached for one width, so a streaming answer re-wraps only itself.
type entry struct {
	kind  entryKind
	text  string
	width int // width of lines; 0: not wrapped yet
	lines []line
}

// segment is a run of text drawn in one role.
type segment struct {
	text string
	role role
}

type line []segment

// wrapped returns the entry's lines for a conversation width.
func (e *entry) wrapped(width int) []line {
	if e.width != width || e.lines == nil {
		e.lines, e.width = entryLines(e.kind, e.text, width), width
	}
	return e.lines
}

func (e *entry) setText(text string) {
	e.text, e.lines = text, nil
}

// entryLines renders one entry; a label hangs over its wrapped text.
func entryLines(kind entryKind, text string, width int) []line {
	label, labelRole, textRole := "", roleInfo, roleInfo
	switch kind {
	case entryUser:
		label, labelRole, textRole = userLabel, roleUser, roleText
	case entryAI:
		label, labelRole, textRole = aiLabel, roleAI, roleText
	case entryError:
		labelRole, textRole = roleError, roleError
	}
	n := len([]rune(label))
	body := WrapParagraphs(text, max(1, width-n))
	lines := make([]line, len(body))
	for i, t := range body {
		prefix := label
		if i > 0 {
			prefix = strings.Repeat(" ", n)
		}
		if prefix != "" {
			lines[i] = append(lines[i], segment{prefix, labelRole})
		}
		lines[i] = append(lines[i], segment{t, textRole})
	}
	return lines
}

// WrapParagraphs wraps text to width cells (docs/tui-spec.md, section 10):
// explicit line breaks and blank lines are kept, tabs expand to stops of
// 4, each line keeps its leading indentation (at most width/2) on all its
// wrapped lines, words are filled greedily and a word longer than the
// width is broken. It follows Python's textwrap.wrap, which the reference
// uses.
func WrapParagraphs(text string, width int) []string {
	width = max(width, 1)
	var out []string
	for _, paragraph := range splitLines(expandTabs(text, 4)) {
		stripped := strings.TrimLeft(paragraph, " ")
		indent := strings.Repeat(" ", min(len(paragraph)-len(stripped), width/2))
		wrapped := wrapLine(stripped, width, indent)
		if len(wrapped) == 0 {
			wrapped = []string{""}
		}
		out = append(out, wrapped...)
	}
	return out
}

// splitLines splits at \n, \r\n and \r; a final line break adds no empty
// line, and "" is one empty line.
func splitLines(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	lines := strings.Split(text, "\n")
	if len(lines) > 1 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

// expandTabs replaces each tab with spaces up to the next multiple of
// size columns; columns restart after each line break.
func expandTabs(text string, size int) string {
	if !strings.Contains(text, "\t") {
		return text
	}
	var b strings.Builder
	col := 0
	for _, r := range text {
		switch r {
		case '\t':
			n := size - col%size
			b.WriteString(strings.Repeat(" ", n))
			col += n
		case '\n', '\r':
			b.WriteRune(r)
			col = 0
		default:
			b.WriteRune(r)
			col++
		}
	}
	return b.String()
}

// wrapLine is textwrap.wrap for one line without leading spaces: chunks
// are runs of whitespace or of other characters; whitespace at the start
// (after the first line) and end of a wrapped line is dropped.
func wrapLine(text string, width int, indent string) []string {
	chunks := splitChunks(text)
	avail := width - len([]rune(indent))
	var lines []string
	for len(chunks) > 0 {
		var cur [][]rune
		curLen := 0
		if len(lines) > 0 && isSpace(chunks[0]) {
			chunks = chunks[1:]
		}
		for len(chunks) > 0 && curLen+len(chunks[0]) <= avail {
			cur = append(cur, chunks[0])
			curLen += len(chunks[0])
			chunks = chunks[1:]
		}
		if len(chunks) > 0 && len(chunks[0]) > avail {
			// A word longer than the width fills the rest of the line.
			end := avail - curLen
			if avail < 1 {
				end = 1
			}
			if end > 0 {
				cur = append(cur, chunks[0][:end])
				chunks[0] = chunks[0][end:]
			}
		}
		if len(cur) > 0 && isSpace(cur[len(cur)-1]) {
			cur = cur[:len(cur)-1]
		}
		if len(cur) > 0 {
			var b strings.Builder
			b.WriteString(indent)
			for _, c := range cur {
				b.WriteString(string(c))
			}
			lines = append(lines, b.String())
		}
	}
	return lines
}

// splitChunks splits text into alternating runs of whitespace (ASCII
// whitespace, which is turned into spaces) and other characters.
func splitChunks(text string) [][]rune {
	var chunks [][]rune
	var cur []rune
	curSpace := false
	for _, r := range text {
		space := strings.ContainsRune(" \t\n\v\f\r", r)
		if space {
			r = ' '
		}
		if len(cur) > 0 && space != curSpace {
			chunks = append(chunks, cur)
			cur = nil
		}
		cur, curSpace = append(cur, r), space
	}
	if len(cur) > 0 {
		chunks = append(chunks, cur)
	}
	return chunks
}

func isSpace(chunk []rune) bool { return len(chunk) > 0 && chunk[0] == ' ' }

// transcript holds the panel's entries and the scroll position.
type transcript struct {
	entries []*entry
	scroll  int // lines scrolled up from the newest line
}

// lineCount is the number of lines of all entries, with one blank line
// between entries.
func (t *transcript) lineCount(width int) int {
	n := 0
	for i, e := range t.entries {
		if i > 0 {
			n++
		}
		n += len(e.wrapped(width))
	}
	return n
}

// lines renders all entries, separated by blank lines.
func (t *transcript) lines(width int) []line {
	var lines []line
	for i, e := range t.entries {
		if i > 0 {
			lines = append(lines, nil)
		}
		lines = append(lines, e.wrapped(width)...)
	}
	return lines
}

// visibleWindow is the [start, end) range of lines shown when scrolled
// up by scroll lines.
func visibleWindow(total, height, scroll int) (int, int) {
	end := total - scroll
	return max(0, end-height), end
}
