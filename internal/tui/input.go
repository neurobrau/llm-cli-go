package tui

// inputLine is the editable input text (docs/tui-spec.md, section 6). It
// works on runes, so the cursor never splits a multi-byte character; one
// rune is one cell on screen.
type inputLine struct {
	text   []rune
	cursor int // 0 … len(text)
}

func (in *inputLine) String() string { return string(in.text) }

func (in *inputLine) set(text string) {
	in.text = []rune(text)
	in.cursor = len(in.text)
}

func (in *inputLine) clear() { in.text, in.cursor = nil, 0 }

func (in *inputLine) insert(r rune) {
	in.text = append(in.text[:in.cursor], append([]rune{r}, in.text[in.cursor:]...)...)
	in.cursor++
}

func (in *inputLine) backspace() {
	if in.cursor > 0 {
		in.text = append(in.text[:in.cursor-1], in.text[in.cursor:]...)
		in.cursor--
	}
}

func (in *inputLine) delete() {
	if in.cursor < len(in.text) {
		in.text = append(in.text[:in.cursor], in.text[in.cursor+1:]...)
	}
}

func (in *inputLine) left()  { in.cursor = max(0, in.cursor-1) }
func (in *inputLine) right() { in.cursor = min(len(in.text), in.cursor+1) }
func (in *inputLine) home()  { in.cursor = 0 }
func (in *inputLine) end()   { in.cursor = len(in.text) }

// view returns the part of the text shown in a field of width cells and
// the cursor's column in it. The text scrolls horizontally so the cursor,
// which may sit after the last character, stays inside the field.
func (in *inputLine) view(width int) (string, int) {
	width = max(width, 1)
	start := max(0, in.cursor-width+1)
	end := min(len(in.text), start+width)
	return string(in.text[start:end]), in.cursor - start
}
