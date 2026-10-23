// Package storage reads and writes NeuroBräu's local JSON files: saved
// conversations (with their usage records), the active provider/model
// selection and non-secret settings, in the schema-version-1 formats shared
// with the Python edition (docs/tui-spec.md, sections 11, 13 and 15).
//
// Nothing here runs on its own: callers decide when to read or write, so a
// session without history (--no-history) can avoid all history I/O.
package storage

import (
	"errors"
	"path/filepath"
)

const (
	// SchemaVersion is the only supported version of every file format.
	SchemaVersion = 1

	appDirName           = "neurobrau"
	conversationsDirName = "conversations"
	activeFileName       = "active.json"
	settingsFileName     = "settings.json"
	dirMode              = 0o700
)

// Paths are the locations of NeuroBräu's files.
type Paths struct {
	Settings      string // <config>/neurobrau/settings.json
	Conversations string // <state>/neurobrau/conversations
	Active        string // <state>/neurobrau/active.json, outside Conversations
}

// ErrNoHome is returned when a path needs the home directory but it is
// unknown.
var ErrNoHome = errors.New("cannot determine the home directory")

// ResolvePaths applies the XDG rules: $XDG_CONFIG_HOME and $XDG_STATE_HOME
// are used only if they are absolute paths; otherwise ~/.config and
// ~/.local/state. getenv reads the environment (os.Getenv in production)
// and home is the home directory ("" if unknown).
func ResolvePaths(getenv func(string) string, home string) (Paths, error) {
	config, err := xdgBase(getenv("XDG_CONFIG_HOME"), home, ".config")
	if err != nil {
		return Paths{}, err
	}
	state, err := xdgBase(getenv("XDG_STATE_HOME"), home, filepath.Join(".local", "state"))
	if err != nil {
		return Paths{}, err
	}
	stateApp := filepath.Join(state, appDirName)
	return Paths{
		Settings:      filepath.Join(config, appDirName, settingsFileName),
		Conversations: filepath.Join(stateApp, conversationsDirName),
		Active:        filepath.Join(stateApp, activeFileName),
	}, nil
}

// xdgBase returns value if it is absolute (a relative XDG value is invalid
// and ignored), otherwise home/fallback.
func xdgBase(value, home, fallback string) (string, error) {
	if filepath.IsAbs(value) {
		return filepath.Clean(value), nil
	}
	if home == "" {
		return "", ErrNoHome
	}
	return filepath.Join(home, fallback), nil
}
