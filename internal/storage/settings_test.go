package storage

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestSettingsDefaults(t *testing.T) {
	_, paths := testPaths(t)
	s := NewSettingsStore(paths.Settings)
	got, warnings := s.Load()
	if !reflect.DeepEqual(got, DefaultSettings()) || got.AnswerStyle != "normal" || warnings != nil || !s.Writable() {
		t.Errorf("missing file: %+v, %q, writable %v", got, warnings, s.Writable())
	}
	if _, err := os.Stat(filepath.Dir(paths.Settings)); err == nil {
		t.Error("Load created the settings directory")
	}
}

func TestSettingsRoundTrip(t *testing.T) {
	_, paths := testPaths(t)
	s := NewSettingsStore(paths.Settings)
	settings := DefaultSettings()
	settings.AnswerStyle = "detailed"
	settings = settings.WithModel("ollama", "qwen3.5:4b", ModelPreferences{Reasoning: "on", MaxOutputTokensSet: true, MaxOutputTokens: 600})
	settings = settings.WithModel("ollama", "hf.co/org/model:Q4_K_M", ModelPreferences{Reasoning: "high"})
	settings = settings.WithModel("openai", "gpt-6-luna", ModelPreferences{MaxOutputTokensSet: true}) // null: model default
	settings = settings.WithModel("openai", "gpt-5", ModelPreferences{Reasoning: ReasoningDefault})
	settings = settings.WithModel("openai", "unused", ModelPreferences{}) // empty: omitted
	if err := s.Save(settings); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(paths.Settings)
	want := `{
  "schema_version": 1,
  "answer_style": "detailed",
  "providers": {
    "ollama": {
      "models": {
        "hf.co/org/model:Q4_K_M": {
          "reasoning": "high"
        },
        "qwen3.5:4b": {
          "reasoning": "on",
          "max_output_tokens": 600
        }
      }
    },
    "openai": {
      "models": {
        "gpt-5": {
          "reasoning": "default"
        },
        "gpt-6-luna": {
          "max_output_tokens": null
        }
      }
    }
  }
}
`
	if string(data) != want {
		t.Errorf("settings.json:\n%s\nwant:\n%s", data, want)
	}
	if perm(t, paths.Settings) != 0o600 || perm(t, filepath.Dir(paths.Settings)) != 0o700 {
		t.Errorf("modes %o / %o", perm(t, paths.Settings), perm(t, filepath.Dir(paths.Settings)))
	}

	loaded, warnings := NewSettingsStore(paths.Settings).Load()
	delete(settings.Models, ModelKey{"openai", "unused"})
	if warnings != nil || !reflect.DeepEqual(loaded, settings) {
		t.Errorf("loaded %+v, %q; want %+v", loaded, warnings, settings)
	}
	luna := loaded.ForModel("openai", "gpt-6-luna")
	if !luna.MaxOutputTokensSet || luna.MaxOutputTokens != 0 || luna.Reasoning != "" {
		t.Errorf("null max_output_tokens: %+v", luna)
	}
	if absent := loaded.ForModel("openai", "gpt-5"); absent.MaxOutputTokensSet {
		t.Errorf("absent max_output_tokens read as set: %+v", absent)
	}
}

func TestSettingsWithEmptyModels(t *testing.T) {
	data, err := EncodeSettings(Settings{})
	want := "{\n  \"schema_version\": 1,\n  \"answer_style\": \"normal\",\n  \"providers\": {}\n}\n"
	if err != nil || string(data) != want {
		t.Errorf("got %q, %v; want %q", data, err, want)
	}
}

func TestSettingsSkipInvalidEntries(t *testing.T) {
	data := `{"schema_version": 1, "answer_style": "chatty", "theme": "dark",
	  "providers": {
	    "openai": {"models": {
	      "gpt-6-luna": {"reasoning": "none", "max_output_tokens": 800, "note": "x"},
	      "gpt-5": {"reasoning": ""},
	      "gpt-5.1": {"max_output_tokens": 0},
	      "gpt-5.2": {"max_output_tokens": -3},
	      "o3": {"max_output_tokens": 1.5},
	      "o4-mini": {"max_output_tokens": true},
	      "gpt-4o": {"reasoning": 5},
	      "gpt-4.1": "fast",
	      "gpt-4.1-mini": {"reasoning": null}
	    }},
	    "ollama": {"models": "none"},
	    "future": {"other": 1}
	  }}`
	s, warnings, err := DecodeSettings([]byte(data))
	if err != nil {
		t.Fatal(err)
	}
	want := DefaultSettings()
	want.Models[ModelKey{"openai", "gpt-6-luna"}] = ModelPreferences{Reasoning: "none", MaxOutputTokensSet: true, MaxOutputTokens: 800}
	want.Models[ModelKey{"openai", "gpt-4.1-mini"}] = ModelPreferences{} // null reasoning: not set
	if !reflect.DeepEqual(s, want) {
		t.Errorf("settings = %+v, want %+v", s, want)
	}
	wantWarnings := []string{
		"Ignoring unknown answer_style 'chatty'; using normal.",
		"Ignoring settings for provider 'future': 'models' must be an object.",
		"Ignoring settings for provider 'ollama': 'models' must be an object.",
		"Ignoring settings for openai/gpt-4.1: entry must be an object.",
		"Ignoring settings for openai/gpt-4o: 'reasoning' must be a non-empty string.",
		"Ignoring settings for openai/gpt-5: 'reasoning' must be a non-empty string.",
		"Ignoring settings for openai/gpt-5.1: 'max_output_tokens' must be a positive integer or null.",
		"Ignoring settings for openai/gpt-5.2: 'max_output_tokens' must be a positive integer or null.",
		"Ignoring settings for openai/o3: 'max_output_tokens' must be a positive integer or null.",
		"Ignoring settings for openai/o4-mini: 'max_output_tokens' must be a positive integer or null.",
	}
	if !reflect.DeepEqual(warnings, wantWarnings) {
		t.Errorf("warnings:\n%s\nwant:\n%s", strings.Join(warnings, "\n"), strings.Join(wantWarnings, "\n"))
	}
}

func TestCorruptSettingsAreNeverOverwritten(t *testing.T) {
	tests := map[string]string{
		"malformed JSON": `{"schema_version": 1, "answer_style": `,
		"schema 2":       `{"schema_version": 2}`,
		"providers list": `{"schema_version": 1, "providers": []}`,
		"providers null": `{"schema_version": 1, "providers": null}`,
		"invalid UTF-8":  "{\"schema_version\": 1, \"answer_style\": \"\xff\"}",
	}
	for name, content := range tests {
		t.Run(name, func(t *testing.T) {
			_, paths := testPaths(t)
			writeFile(t, paths.Settings, content)
			s := NewSettingsStore(paths.Settings)
			got, warnings := s.Load()
			if !reflect.DeepEqual(got, DefaultSettings()) || len(warnings) != 1 ||
				!strings.HasPrefix(warnings[0], "Ignoring invalid settings file "+paths.Settings+": ") ||
				!strings.HasSuffix(warnings[0], "Using defaults; changes will not be saved until the file is fixed or removed.") {
				t.Errorf("got %+v, %q", got, warnings)
			}
			if s.Writable() {
				t.Error("store still writable")
			}
			err := s.Save(DefaultSettings())
			if err == nil || err.Error() != "Settings not saved: "+paths.Settings+" could not be read at startup." {
				t.Errorf("Save: %v", err)
			}
			if data, _ := os.ReadFile(paths.Settings); string(data) != content {
				t.Error("the invalid file was overwritten")
			}
		})
	}
	t.Run("unreadable", func(t *testing.T) {
		_, paths := testPaths(t)
		if err := os.MkdirAll(paths.Settings, 0o700); err != nil { // a directory where the file should be
			t.Fatal(err)
		}
		s := NewSettingsStore(paths.Settings)
		_, warnings := s.Load()
		want := "Could not read settings at " + paths.Settings + ": Is a directory. Using defaults; changes will not be saved."
		if len(warnings) != 1 || warnings[0] != want || s.Writable() {
			t.Errorf("warnings = %q, writable %v; want %q", warnings, s.Writable(), want)
		}
	})
}

func TestSettingsOnlyTightensTheAppDirectory(t *testing.T) {
	home, paths := testPaths(t)
	config := filepath.Join(home, ".config")
	if err := os.MkdirAll(filepath.Join(config, "neurobrau"), 0o755); err != nil {
		t.Fatal(err)
	}
	os.Chmod(config, 0o755)
	os.Chmod(filepath.Join(config, "neurobrau"), 0o755)
	if err := NewSettingsStore(paths.Settings).Save(DefaultSettings()); err != nil {
		t.Fatal(err)
	}
	if perm(t, config) != 0o755 || perm(t, filepath.Join(config, "neurobrau")) != 0o700 {
		t.Errorf("config %o, app %o", perm(t, config), perm(t, filepath.Join(config, "neurobrau")))
	}
	assertNoTempFiles(t, filepath.Dir(paths.Settings))
}
