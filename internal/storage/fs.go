package storage

import (
	"bytes"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unicode"
	"unicode/utf8"
)

// syncFile flushes a file to disk; tests replace it to simulate failures.
var syncFile = (*os.File).Sync

// writeAtomically replaces path so readers see either the old or the new
// content, never a mix: a temp file `.<name>.<random>.tmp` (mode 0600) in
// the same directory is written, synced and renamed over path, then the
// directory is synced where supported. The temp file is removed on error.
func writeAtomically(path string, data []byte) (err error) {
	dir, name := filepath.Split(path)
	stem := strings.TrimSuffix(name, filepath.Ext(name))
	tmp, err := os.CreateTemp(dir, "."+stem+".*.tmp") // created with mode 0600
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			tmp.Close()
			os.Remove(tmp.Name())
		}
	}()
	if _, err = tmp.Write(data); err != nil {
		return err
	}
	if err = syncFile(tmp); err != nil {
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	if err = os.Rename(tmp.Name(), path); err != nil {
		return err
	}
	syncDir(dir)
	return nil
}

// syncDir persists a rename; best effort, as not every platform supports it.
func syncDir(dir string) {
	if d, err := os.Open(dir); err == nil {
		d.Sync()
		d.Close()
	}
}

// ensureDir creates dir with mode 0700 and tightens it (and owned, the app
// directories it belongs to) to 0700. Missing ancestors above the app
// directory are created like `mkdir -p`; existing ones are never changed.
func ensureDir(dir string, owned []string) error {
	if len(owned) > 0 {
		if err := os.MkdirAll(filepath.Dir(owned[0]), 0o777); err != nil {
			return err
		}
	}
	if err := os.MkdirAll(dir, dirMode); err != nil {
		return err
	}
	for _, d := range owned {
		if err := os.Chmod(d, dirMode); err != nil { // mkdir's mode is reduced by the umask
			return err
		}
	}
	return nil
}

// appDirs returns dir and, if its parent is the app directory "neurobrau",
// that parent too (outermost first). Directories above are not ours.
func appDirs(dir string) []string {
	if parent := filepath.Dir(dir); filepath.Base(parent) == appDirName {
		return []string{parent, dir}
	}
	return []string{dir}
}

// secureExistingDirs tightens existing app directories to 0700; best
// effort, as a problem shows up later when reading or saving.
func secureExistingDirs(dirs []string) {
	for _, d := range dirs {
		if info, err := os.Stat(d); err == nil && info.IsDir() {
			os.Chmod(d, dirMode)
		}
	}
}

// reason describes an I/O error for users, without a stack trace:
// "Permission denied", "No such file or directory", …
func reason(err error) string {
	var errno syscall.Errno
	if errors.As(err, &errno) {
		return capitalize(errno.Error())
	}
	var pathErr *fs.PathError
	if errors.As(err, &pathErr) {
		return capitalize(pathErr.Err.Error())
	}
	return capitalize(err.Error())
}

func capitalize(s string) string {
	r, size := utf8.DecodeRuneInString(s)
	if size == 0 {
		return s
	}
	return string(unicode.ToUpper(r)) + s[size:]
}

// marshalIndent encodes v like Python's json.dumps(v, ensure_ascii=False,
// indent=2): two-space indentation, non-ASCII and <>& as is, and a trailing
// newline.
func marshalIndent(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil { // Encode adds the trailing newline
		return nil, err
	}
	return buf.Bytes(), nil
}
