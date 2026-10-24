"""Development helper: Python 0.3.0 <-> Go generation-settings resolution parity.

Uses the read-only Python reference implementation; it never writes into
it (bytecode caching is disabled) and reads only the file given.

  python3 -B scripts/python_settings_parity.py <python-src> <cases.json>

<cases.json> is a list of cases: a model's reasoning support (or a lookup
error) and output limit, saved per-model preferences, command-line
overrides and strictness. For each, Python's
preferences.resolve_model_settings result is printed as JSON on stdout.
Not used by `go test ./...` unless NEUROBRAU_PYTHON_SRC is set (see
internal/chat/python_parity_test.go).
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main(src: str, cases_path: Path) -> None:
    sys.path.insert(0, src)
    from neurobrau_cli import preferences  # noqa: PLC0415
    from neurobrau_cli.providers.base import ErrorKind, ProviderError, ReasoningSupport  # noqa: PLC0415
    from neurobrau_cli.settings import ModelPreferences, SettingsError  # noqa: PLC0415

    class FakeProvider:
        def __init__(self, case: dict) -> None:
            self.case = case

        def reasoning_support(self, model: str) -> ReasoningSupport:
            if self.case["support_error"]:
                raise ProviderError(self.case["support_error"], ErrorKind.CONNECTION)
            s = self.case["support"]
            return ReasoningSupport(tuple(s["values"]), s["known"], s["default"], s["allows_default"])

        def max_output_limit(self, model: str) -> int | None:
            return self.case["limit"]

    result = []
    for case in json.loads(cases_path.read_text(encoding="utf-8")):
        prefs = ModelPreferences(**case["prefs"])
        overrides = preferences.Overrides(**case["overrides"])
        try:
            resolved, notices = preferences.resolve_model_settings(
                FakeProvider(case), "m", prefs, overrides, case["strict"])
        except SettingsError as error:
            result.append({"error": str(error)})
            continue
        result.append({
            "reasoning": resolved.reasoning,
            "explicit": resolved.reasoning_explicit,
            "max_output_tokens": resolved.max_output_tokens,
            "notices": notices,
        })
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
