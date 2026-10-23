package storage

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"unicode/utf8"
)

// object is a JSON object whose fields are checked one by one, so the
// validation can tell a missing field from null and from a wrong type, as
// the Python reference does.
type object map[string]json.RawMessage

// decodeObject parses a whole file. Invalid UTF-8 is an error (Go's decoder
// would otherwise replace it silently), and the top level must be an object.
func decodeObject(data []byte) (object, error) {
	if !utf8.Valid(data) {
		return nil, errors.New("not valid UTF-8")
	}
	var v json.RawMessage
	if err := json.Unmarshal(data, &v); err != nil {
		return nil, fmt.Errorf("not valid JSON (%v)", err)
	}
	o, ok := asObject(v)
	if !ok {
		return nil, errors.New("top level must be a JSON object")
	}
	return o, nil
}

// asObject returns raw as an object; false for null and other types.
func asObject(raw json.RawMessage) (object, bool) {
	var o object
	if kind(raw) != '{' || json.Unmarshal(raw, &o) != nil {
		return nil, false
	}
	return o, true
}

// kind returns the first character of a JSON value: '{', '[', '"', 't',
// 'f', 'n' (null) or a digit/'-' for numbers; 0 if raw is empty (missing).
func kind(raw json.RawMessage) byte {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return 0
	}
	return raw[0]
}

// present reports whether key exists and is not null.
func (o object) present(key string) bool {
	k := kind(o[key])
	return k != 0 && k != 'n'
}

// str returns a string field; ok is false if it is missing, null or not a
// string.
func (o object) str(key string) (string, bool) {
	var s string
	if kind(o[key]) != '"' || json.Unmarshal(o[key], &s) != nil {
		return "", false
	}
	return s, true
}

// count returns a non-negative integer field. A missing or null field is
// (0, false, nil); anything else that is not an integer ≥ 0 (booleans,
// fractions, strings, negative numbers) is an error.
func (o object) count(key string) (int, bool, error) {
	raw := o[key]
	if !o.present(key) {
		return 0, false, nil
	}
	var n int
	if k := kind(raw); k == 't' || k == 'f' || json.Unmarshal(raw, &n) != nil || n < 0 {
		return 0, false, fmt.Errorf("'%s' must be a non-negative integer", key)
	}
	return n, true, nil
}

// pyRepr formats a JSON value roughly like Python's repr() of the parsed
// value, for messages shared with the Python edition: 'text', None, True.
func pyRepr(raw json.RawMessage) string {
	switch kind(raw) {
	case 0, 'n':
		return "None"
	case 't':
		return "True"
	case 'f':
		return "False"
	case '"':
		var s string
		json.Unmarshal(raw, &s)
		return "'" + s + "'"
	}
	return string(bytes.TrimSpace(raw))
}
