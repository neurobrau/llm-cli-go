"""Development helper: Python 0.3.0 <-> Go cost estimate parity.

Uses the read-only Python reference implementation; it never writes into
it (bytecode caching is disabled) and reads only the file given.

  python3 -B scripts/python_pricing_parity.py <python-src> <cases.json>

<cases.json> maps a case name to a list of usage records in the
conversation-file format (docs/tui-spec.md, section 15). For each case,
the records are parsed with the Python readers (malformed ones skipped)
and Python's estimates, panel cost and /usage report are printed as JSON
on stdout. Not used by `go test ./...` unless NEUROBRAU_PYTHON_SRC is set
(see internal/chat/python_parity_test.go).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True


def main(src: str, cases_path: Path) -> None:
    sys.path.insert(0, src)
    from neurobrau_cli import pricing, storage  # noqa: PLC0415
    from neurobrau_cli.chat import usage_report  # noqa: PLC0415

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    result = {}
    for name, raw in cases.items():
        records = storage.parse_requests(raw)
        total = pricing.total_cost(records)
        result[name] = {
            "estimates": [
                {
                    "status": e.status,
                    "amount": None if e.amount is None else str(e.amount),
                    "notes": list(e.notes),
                }
                for e in map(pricing.estimate, records)
            ],
            "short": pricing.format_cost_short(total),
            "describe": pricing.describe_cost(total),
            "report": usage_report(SimpleNamespace(requests=list(records))),
        }
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
