package storage

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// HistoryStore reads and writes the conversation files in one directory
// and the active selection next to it. It does nothing until a method is
// called.
type HistoryStore struct {
	dir string
}

// NewHistoryStore returns a store for the conversations directory dir
// (normally Paths.Conversations). active.json lives in its parent.
func NewHistoryStore(dir string) *HistoryStore { return &HistoryStore{dir: dir} }

// Dir returns the conversations directory.
func (s *HistoryStore) Dir() string { return s.dir }

// PathFor returns the file of a conversation.
func (s *HistoryStore) PathFor(id string) string { return filepath.Join(s.dir, id+".json") }

// ActivePath returns active.json, next to the conversations directory so
// that cleanup never removes it.
func (s *HistoryStore) ActivePath() string { return filepath.Join(filepath.Dir(s.dir), activeFileName) }

// historyError describes a failed history operation for users.
func historyError(action, path string, err error) error {
	return fmt.Errorf("Could not %s conversation history at %s: %s.", action, path, reason(err))
}

// ListPaths returns the `*.json` entries of the directory, sorted by name.
// A missing directory has none. Other files are ignored.
func (s *HistoryStore) ListPaths() ([]string, error) {
	entries, err := os.ReadDir(s.dir)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, historyError("read", s.dir, err)
	}
	var paths []string
	for _, e := range entries { // ReadDir sorts by name
		if strings.HasSuffix(e.Name(), ".json") {
			paths = append(paths, filepath.Join(s.dir, e.Name()))
		}
	}
	return paths, nil
}

// Load reads and validates one conversation file. It never modifies it.
func (s *HistoryStore) Load(path string) (Conversation, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Conversation{}, historyError("read", path, err)
	}
	c, err := DecodeConversation(data)
	if err != nil {
		return Conversation{}, fmt.Errorf("Ignoring unreadable conversation file %s: %v. The file was left unchanged.", path, err)
	}
	return c, nil
}

// LoadLatest returns the valid conversation with the greatest updated_at
// among those match accepts (nil: all), with ties broken by the newest
// file modification time; found is false if there is none. Invalid files
// are reported as warnings and left unchanged. No file is written; existing
// app directories are tightened to 0700, as at every startup.
func (s *HistoryStore) LoadLatest(match func(Conversation) bool) (latest Conversation, found bool, warnings []string) {
	secureExistingDirs(appDirs(s.dir))
	paths, err := s.ListPaths()
	if err != nil {
		return Conversation{}, false, []string{err.Error()}
	}
	var latestMod time.Time
	for _, path := range paths {
		c, err := s.Load(path)
		if err != nil {
			warnings = append(warnings, err.Error())
			continue
		}
		info, err := os.Stat(path)
		if err != nil {
			warnings = append(warnings, historyError("read", path, err).Error())
			continue
		}
		if match != nil && !match(c) {
			continue
		}
		newer := c.UpdatedAt.After(latest.UpdatedAt) ||
			c.UpdatedAt.Equal(latest.UpdatedAt) && info.ModTime().After(latestMod)
		if !found || newer {
			latest, latestMod, found = c, info.ModTime(), true
		}
	}
	return latest, found, warnings
}

// Save writes a conversation atomically to `<id>.json` (mode 0600),
// creating the directories with mode 0700.
func (s *HistoryStore) Save(c Conversation) error {
	path := s.PathFor(c.ID)
	data, err := EncodeConversation(c)
	if err != nil {
		return fmt.Errorf("Could not save conversation history at %s: %v.", path, err)
	}
	if err := ensureDir(s.dir, appDirs(s.dir)); err != nil {
		return historyError("save", path, err)
	}
	if err := writeAtomically(path, data); err != nil {
		return historyError("save", path, err)
	}
	return nil
}

// CountConversations returns the number of `*.json` entries, e.g. to ask
// for confirmation before DeleteAll.
func (s *HistoryStore) CountConversations() (int, error) {
	paths, err := s.ListPaths()
	return len(paths), err
}

// DeleteAll deletes every `*.json` file in the conversations directory and
// returns how many were deleted. Only call it after the user confirmed.
// Other files, subdirectories, active.json and settings are left alone.
func (s *HistoryStore) DeleteAll() (int, error) {
	paths, err := s.ListPaths()
	if err != nil {
		return 0, err
	}
	deleted := 0
	for _, path := range paths {
		info, err := os.Lstat(path)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err == nil && info.IsDir() {
			continue // not a conversation file
		}
		if err == nil {
			err = os.Remove(path)
		}
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return deleted, fmt.Errorf("Deleted %d file(s), then failed: %w", deleted, historyError("delete", path, err))
		}
		deleted++
	}
	return deleted, nil
}

// Selection is a provider and model.
type Selection struct {
	Provider string
	Model    string
}

// knownProviders can appear in active.json.
var knownProviders = []string{ProviderOpenAI, ProviderOllama}

// LoadActive returns the selection saved in active.json. A missing file
// gives found == false and no warning; an unreadable or invalid one (also
// an unknown provider) gives a warning and is left unchanged.
func (s *HistoryStore) LoadActive() (sel Selection, found bool, warning string) {
	path := s.ActivePath()
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return Selection{}, false, ""
	}
	if err != nil {
		return Selection{}, false, fmt.Sprintf("Ignoring unreadable %s: %s.", path, reason(err))
	}
	o, err := decodeObject(data)
	if err != nil && !json.Valid(data) { // not UTF-8 or not JSON at all
		return Selection{}, false, fmt.Sprintf("Ignoring unreadable %s: %v.", path, err)
	}
	provider, _ := o.str("provider")
	model, _ := o.str("model")
	if checkSchemaVersion(o) != nil || !slices.Contains(knownProviders, provider) || model == "" {
		return Selection{}, false, fmt.Sprintf("Ignoring invalid %s.", path)
	}
	return Selection{Provider: provider, Model: model}, true, ""
}

// SaveActive writes active.json atomically (mode 0600) in the same
// one-line format as the Python edition. Conversation files are not
// touched.
func (s *HistoryStore) SaveActive(sel Selection) error {
	path := s.ActivePath()
	if !slices.Contains(knownProviders, sel.Provider) || sel.Model == "" {
		return fmt.Errorf("Could not save conversation history at %s: invalid selection %q/%q.", path, sel.Provider, sel.Model)
	}
	data := fmt.Appendf(nil, `{"schema_version": %d, "provider": %s, "model": %s}`+"\n",
		SchemaVersion, jsonString(sel.Provider), jsonString(sel.Model))
	if err := ensureDir(s.dir, appDirs(s.dir)); err != nil {
		return historyError("save", path, err)
	}
	if err := writeAtomically(path, data); err != nil {
		return historyError("save", path, err)
	}
	return nil
}

// jsonString encodes s as a JSON string without escaping <, > and &.
func jsonString(s string) string {
	data, _ := marshalIndent(s)
	return strings.TrimSuffix(string(data), "\n")
}
