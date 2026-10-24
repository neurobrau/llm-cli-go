package tui

import (
	"reflect"
	"strings"
	"testing"
)

// The expected lines are the output of the Python reference's
// wrap_paragraphs (textwrap.wrap) for the same input.
func TestWrapParagraphsMatchesPython(t *testing.T) {
	tests := []struct {
		text  string
		width int
		want  []string
	}{
		{"A neural network is a model made of layers of connected units that learn weights from data.", 50,
			[]string{"A neural network is a model made of layers of", "connected units that learn weights from data."}},
		{"", 10, []string{""}},
		{"one\n\ntwo\n", 10, []string{"one", "", "two"}},
		{"trailing\n\n", 10, []string{"trailing", ""}},
		{"abcdefghijklmnopqrstuvwxyz", 10, []string{"abcdefghij", "klmnopqrst", "uvwxyz"}},
		{"ab verylongword more", 10, []string{"ab verylon", "gword more"}},
		{"abcd efghijklmnop", 5, []string{"abcd", "efghi", "jklmn", "op"}},
		{"    indented code line that wraps around", 20, []string{"    indented code", "    line that wraps", "    around"}},
		{"\tx = 1\n\t\ty\nab\tc", 20, []string{"    x = 1", "        y", "ab  c"}},
		{"                    deep indent text here", 10, []string{"     deep", "     inden", "     t", "     text", "     here"}},
		{"Grüß Gott, Привет мир, this wraps", 12, []string{"Grüß Gott,", "Привет мир,", "this wraps"}},
		{"a  b   c", 3, []string{"a", "b", "c"}},
		{"well-known hyphenated-words stay", 12, []string{"well-known h", "yphenated-wo", "rds stay"}},
		{"   ", 10, []string{""}},
		{"x\r\ny\rz", 10, []string{"x", "y", "z"}},
		{strings.Repeat("word ", 5), 9, []string{"word word", "word word", "word"}},
	}
	for _, tt := range tests {
		if got := WrapParagraphs(tt.text, tt.width); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("WrapParagraphs(%q, %d) = %q, want %q", tt.text, tt.width, got, tt.want)
		}
	}
}

func TestWrapNeverExceedsWidth(t *testing.T) {
	text := "Здравствуйте! Ein sehr langes Wort: Donaudampfschifffahrtsgesellschaftskapitän.\n\n\tcode()\n   x"
	for width := 1; width <= 30; width++ {
		for _, l := range WrapParagraphs(text, width) {
			if n := len([]rune(l)); n > width {
				t.Fatalf("width %d: line %q has %d cells", width, l, n)
			}
		}
	}
}

func TestEntryLinesHangingIndent(t *testing.T) {
	lines := entryLines(entryAI, "A neural network is a model made of layers of connected units that learn weights from data.", 55)
	var got []string
	for _, l := range lines {
		var b strings.Builder
		for _, s := range l {
			b.WriteString(s.text)
		}
		got = append(got, b.String())
	}
	want := []string{"AI > A neural network is a model made of layers of", "     connected units that learn weights from data."}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q", got)
	}
	if lines[0][0].role != roleAI || lines[0][1].role != roleText || lines[1][0].text != "     " {
		t.Errorf("roles %+v", lines)
	}
	// Info and error entries have no label and use the full width.
	if l := entryLines(entryError, "Error: boom", 20); len(l) != 1 || len(l[0]) != 1 || l[0][0].role != roleError {
		t.Errorf("error entry %+v", l)
	}
}

func TestTranscriptLinesAndWindow(t *testing.T) {
	tr := transcript{entries: []*entry{{kind: entryUser, text: "Hi"}, {kind: entryAI, text: "one\ntwo"}, {kind: entryInfo, text: "note"}}}
	if n := tr.lineCount(40); n != 6 { // 1 + blank + 2 + blank + 1
		t.Errorf("lineCount %d", n)
	}
	if len(tr.lines(40)[1]) != 0 {
		t.Error("no blank line between entries")
	}
	for _, tt := range []struct{ total, height, scroll, start, end int }{
		{6, 4, 0, 2, 6}, {6, 4, 2, 0, 4}, {3, 4, 0, 0, 3},
	} {
		if s, e := visibleWindow(tt.total, tt.height, tt.scroll); s != tt.start || e != tt.end {
			t.Errorf("visibleWindow(%d,%d,%d) = %d,%d", tt.total, tt.height, tt.scroll, s, e)
		}
	}
}

func TestEntryCacheFollowsTextAndWidth(t *testing.T) {
	e := &entry{kind: entryAI, text: "short"}
	if len(e.wrapped(40)) != 1 {
		t.Fatal("one line")
	}
	e.setText(strings.Repeat("word ", 20))
	if len(e.wrapped(40)) < 2 {
		t.Error("setText did not re-wrap")
	}
	if len(e.wrapped(10)) <= len(e.wrapped(40)) {
		t.Error("a narrower width did not re-wrap")
	}
}
