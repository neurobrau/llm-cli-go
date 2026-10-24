package llm

import "testing"

func TestResolveReasoning(t *testing.T) {
	luna := ReasoningSupport{Values: []string{"none", "low", "medium", "high", "xhigh", "max"}, Known: true, Default: "none"}
	gpt5 := ReasoningSupport{Values: []string{"minimal", "low", "medium", "high"}, Known: true}
	noReasoning := ReasoningSupport{Known: true}
	unknown := ReasoningSupport{}
	onOff := ReasoningSupport{Values: []string{"off", "on"}, Known: true, Default: "off", NoModelDefault: true}
	levels := ReasoningSupport{Values: []string{"low", "medium", "high"}, Known: true}

	tests := []struct {
		name      string
		support   ReasoningSupport
		requested string
		want      Reasoning
		wantErr   string
	}{
		{"no flag", luna, "", Reasoning{}, ""},
		{"model default", luna, "model-default", Reasoning{Explicit: true}, ""},
		{"supported effort", luna, "high", Reasoning{Effort: "high", Explicit: true}, ""},
		{"explicit none", luna, "none", Reasoning{Effort: "none", Explicit: true}, ""},
		{"unsupported effort", gpt5, "none", Reasoning{}, "m supports reasoning: Minimal, Low, Medium, High; 'none' is not available."},
		{"xhigh label", ReasoningSupport{Values: []string{"xhigh"}, Known: true}, "low", Reasoning{}, "m supports reasoning: XHigh; 'low' is not available."},
		{"none without reasoning", noReasoning, "none", Reasoning{Explicit: true}, ""},
		{"off without reasoning", noReasoning, "off", Reasoning{Explicit: true}, ""}, // as in the Python edition
		{"effort without reasoning", noReasoning, "high", Reasoning{}, "m does not support reasoning; 'high' is not available."},
		{"unknown model passes through", unknown, "max", Reasoning{Effort: "max", Explicit: true}, ""},
		{"unknown model, model default", unknown, "model-default", Reasoning{Explicit: true}, ""},
		{"on/off: none means off", onOff, "none", Reasoning{Effort: "off", Explicit: true}, ""},
		{"on/off: effort means on", onOff, "low", Reasoning{Effort: "on", Explicit: true}, ""},
		{"on/off: no model default", onOff, "model-default", Reasoning{}, "m has no model-default reasoning choice (m supports reasoning: Off, On)."},
		{"levels: named level", levels, "high", Reasoning{Effort: "high", Explicit: true}, ""},
		{"levels: no off", levels, "none", Reasoning{}, "m supports reasoning: Low, Medium, High; 'none' is not available."},
		{"levels: model default", levels, "model-default", Reasoning{Explicit: true}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolveReasoning("m", tt.support, tt.requested)
			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil || got != tt.want {
				t.Errorf("got %+v, %v; want %+v", got, err, tt.want)
			}
		})
	}
}

func TestCheckMaxOutputTokens(t *testing.T) {
	tests := []struct {
		value, limit int
		wantErr      string
	}{
		{256, 0, ""},
		{16384, 16384, ""},
		{16385, 16384, "16385 exceeds the limit of 16384 output tokens for m."},
		{0, 0, "Max output tokens must be a positive whole number, not 0."},
	}
	for _, tt := range tests {
		err := CheckMaxOutputTokens("m", tt.value, tt.limit)
		if (tt.wantErr == "" && err != nil) || (tt.wantErr != "" && (err == nil || err.Error() != tt.wantErr)) {
			t.Errorf("CheckMaxOutputTokens(%d, %d) = %v, want %q", tt.value, tt.limit, err, tt.wantErr)
		}
	}
}

func TestStyleInstructions(t *testing.T) {
	// Exact texts of the Python reference (settings.STYLE_INSTRUCTIONS).
	tests := map[string]string{
		"brief": "Answer briefly and concisely. Use a few sentences or a short list. " +
			"Leave out background and caveats unless the user asks for them.",
		"normal": "",
		"detailed": "Give a comprehensive, well-structured explanation. Cover the relevant " +
			"background and the reasoning behind the answer, and include concrete " +
			"examples when they help.",
		"": "",
	}
	for style, want := range tests {
		if got := StyleInstructions(style); got != want {
			t.Errorf("StyleInstructions(%q) = %q, want %q", style, got, want)
		}
	}
}
