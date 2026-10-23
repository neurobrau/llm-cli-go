package storage

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// ReasoningDefault is the saved reasoning value meaning "send no reasoning
// parameter" (the model's own default).
const ReasoningDefault = "default"

// ModelPreferences are the saved choices for one provider/model. Zero
// values mean "not set": built-in defaults apply.
type ModelPreferences struct {
	// Reasoning is a provider value ("none", "low", "off", "on", a level,
	// …), ReasoningDefault, or "" if not set. Whether the model supports it
	// is checked when it is used.
	Reasoning string
	// MaxOutputTokensSet distinguishes a saved choice from "not set". With
	// it, MaxOutputTokens is the limit, or 0 for "the model's default"
	// (JSON null).
	MaxOutputTokensSet bool
	MaxOutputTokens    int
}

func (p ModelPreferences) empty() bool { return p.Reasoning == "" && !p.MaxOutputTokensSet }

// ModelKey identifies a model of a provider. Model IDs are exact and may
// contain "/" and ":".
type ModelKey struct {
	Provider string
	Model    string
}

// Settings are the non-secret preferences in settings.json.
type Settings struct {
	AnswerStyle string // llm.AnswerStyles; default llm.StyleNormal
	Models      map[ModelKey]ModelPreferences
}

// DefaultSettings returns the settings used without a file.
func DefaultSettings() Settings {
	return Settings{AnswerStyle: llm.StyleNormal, Models: map[ModelKey]ModelPreferences{}}
}

// ForModel returns the saved preferences of a model (zero if none).
func (s Settings) ForModel(provider, model string) ModelPreferences {
	return s.Models[ModelKey{provider, model}]
}

// WithModel returns a copy of s with the preferences of one model replaced.
func (s Settings) WithModel(provider, model string, prefs ModelPreferences) Settings {
	s.Models = maps.Clone(s.Models)
	if s.Models == nil {
		s.Models = map[ModelKey]ModelPreferences{}
	}
	s.Models[ModelKey{provider, model}] = prefs
	return s
}

// nullableInt is a JSON field that is written as null or a number, and
// omitted when not set.
type nullableInt struct {
	set   bool
	value int // 0: null
}

func (n nullableInt) IsZero() bool { return !n.set }

func (n nullableInt) MarshalJSON() ([]byte, error) {
	if n.value == 0 {
		return []byte("null"), nil
	}
	return json.Marshal(n.value)
}

type modelEntryJSON struct {
	Reasoning       string      `json:"reasoning,omitempty"`
	MaxOutputTokens nullableInt `json:"max_output_tokens,omitzero"`
}

type providerJSON struct {
	Models map[string]modelEntryJSON `json:"models"`
}

type settingsJSON struct {
	SchemaVersion int                     `json:"schema_version"`
	AnswerStyle   string                  `json:"answer_style"`
	Providers     map[string]providerJSON `json:"providers"`
}

// EncodeSettings returns the file content: providers and models sorted by
// name, empty entries omitted, 2-space indent, trailing newline.
func EncodeSettings(s Settings) ([]byte, error) {
	style := s.AnswerStyle
	if style == "" {
		style = llm.StyleNormal
	}
	if !slices.Contains(llm.AnswerStyles, style) {
		return nil, fmt.Errorf("unknown answer style %q", style)
	}
	out := settingsJSON{SchemaVersion: SchemaVersion, AnswerStyle: style, Providers: map[string]providerJSON{}}
	for key, prefs := range s.Models {
		if prefs.empty() {
			continue
		}
		if prefs.MaxOutputTokens < 0 {
			return nil, fmt.Errorf("invalid max_output_tokens %d for %s/%s", prefs.MaxOutputTokens, key.Provider, key.Model)
		}
		p, ok := out.Providers[key.Provider]
		if !ok {
			p = providerJSON{Models: map[string]modelEntryJSON{}}
			out.Providers[key.Provider] = p
		}
		p.Models[key.Model] = modelEntryJSON{
			Reasoning:       prefs.Reasoning,
			MaxOutputTokens: nullableInt{set: prefs.MaxOutputTokensSet, value: prefs.MaxOutputTokens},
		}
	}
	return marshalIndent(out) // encoding/json writes map keys sorted
}

// DecodeSettings parses settings.json. Invalid individual values are
// skipped with a warning; an error means the file as a whole is unusable.
// Unknown fields are ignored.
func DecodeSettings(data []byte) (Settings, []string, error) {
	o, err := decodeObject(data)
	if err != nil {
		return Settings{}, nil, err
	}
	if err := checkSchemaVersion(o); err != nil {
		return Settings{}, nil, err
	}
	s := DefaultSettings()
	var warnings []string
	if raw, found := o["answer_style"]; found {
		style, _ := o.str("answer_style")
		if slices.Contains(llm.AnswerStyles, style) {
			s.AnswerStyle = style
		} else {
			warnings = append(warnings, fmt.Sprintf("Ignoring unknown answer_style %s; using %s.", pyRepr(raw), llm.StyleNormal))
		}
	}

	providers := object{}
	if raw, found := o["providers"]; found {
		var ok bool
		if providers, ok = asObject(raw); !ok {
			return Settings{}, nil, errors.New("'providers' must be an object")
		}
	}
	for _, provider := range slices.Sorted(maps.Keys(providers)) {
		p, _ := asObject(providers[provider])
		models, ok := asObject(p["models"])
		if !ok {
			warnings = append(warnings, fmt.Sprintf("Ignoring settings for provider '%s': 'models' must be an object.", provider))
			continue
		}
		for _, model := range slices.Sorted(maps.Keys(models)) {
			prefs, problem := parseModelEntry(models[model])
			if problem != "" {
				warnings = append(warnings, fmt.Sprintf("Ignoring settings for %s/%s: %s.", provider, model, problem))
				continue
			}
			s.Models[ModelKey{provider, model}] = prefs
		}
	}
	return s, warnings, nil
}

func parseModelEntry(raw json.RawMessage) (ModelPreferences, string) {
	entry, ok := asObject(raw)
	if !ok {
		return ModelPreferences{}, "entry must be an object"
	}
	var prefs ModelPreferences
	if entry.present("reasoning") {
		if prefs.Reasoning, ok = entry.str("reasoning"); !ok || prefs.Reasoning == "" {
			return ModelPreferences{}, "'reasoning' must be a non-empty string"
		}
	}
	if _, found := entry["max_output_tokens"]; found {
		prefs.MaxOutputTokensSet = true
		n, given, err := entry.count("max_output_tokens")
		if err != nil || given && n == 0 {
			return ModelPreferences{}, "'max_output_tokens' must be a positive integer or null"
		}
		prefs.MaxOutputTokens = n
	}
	return prefs, ""
}

// SettingsStore loads and saves settings.json. A file that could not be
// read or parsed is never overwritten during the store's lifetime.
type SettingsStore struct {
	path     string
	writable bool
}

// NewSettingsStore returns a store for path (normally Paths.Settings).
func NewSettingsStore(path string) *SettingsStore {
	return &SettingsStore{path: path, writable: true}
}

// Path returns the settings file.
func (s *SettingsStore) Path() string { return s.path }

// Writable reports whether Save may write the file.
func (s *SettingsStore) Writable() bool { return s.writable }

// Load returns the settings, or the defaults with a warning if the file is
// unusable; a missing file gives the defaults without a warning.
func (s *SettingsStore) Load() (Settings, []string) {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return DefaultSettings(), nil
	}
	if err != nil {
		s.writable = false
		return DefaultSettings(), []string{fmt.Sprintf(
			"Could not read settings at %s: %s. Using defaults; changes will not be saved.", s.path, reason(err))}
	}
	settings, warnings, err := DecodeSettings(data)
	if err != nil {
		s.writable = false
		return DefaultSettings(), []string{fmt.Sprintf(
			"Ignoring invalid settings file %s: %v. Using defaults; "+
				"changes will not be saved until the file is fixed or removed.", s.path, err)}
	}
	return settings, warnings
}

// Save writes the settings atomically (mode 0600; an app directory
// "neurobrau" is created or tightened to 0700). It refuses to overwrite a
// file that could not be loaded.
func (s *SettingsStore) Save(settings Settings) error {
	if !s.writable {
		return fmt.Errorf("Settings not saved: %s could not be read at startup.", s.path)
	}
	data, err := EncodeSettings(settings)
	if err != nil {
		return fmt.Errorf("Could not save settings at %s: %v.", s.path, err)
	}
	dir := filepath.Dir(s.path)
	var owned []string
	if filepath.Base(dir) == appDirName {
		owned = []string{dir}
	}
	if err := ensureDir(dir, owned); err != nil {
		return fmt.Errorf("Could not save settings at %s: %s.", s.path, reason(err))
	}
	if err := writeAtomically(s.path, data); err != nil {
		return fmt.Errorf("Could not save settings at %s: %s.", s.path, reason(err))
	}
	return nil
}
