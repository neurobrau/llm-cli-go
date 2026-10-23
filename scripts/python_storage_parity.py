"""Development helper: Python 0.3.0 <-> Go storage JSON parity.

Uses the read-only Python reference implementation; it never writes into
it (bytecode caching is disabled) and touches only the directories given.

  python3 -B scripts/python_storage_parity.py generate <python-src> <out-dir>
      Write sanitized fixtures with the Python serializers (synthetic data).
  python3 -B scripts/python_storage_parity.py check <python-src> <dir>
      Load Go-written files from <dir> with the Python readers and print
      what Python writes back for each, as JSON on stdout.

<python-src> is ../llm-cli-python/src. Not used by `go test ./...` unless
NEUROBRAU_PYTHON_SRC is set (see internal/storage/python_parity_test.go).
"""

import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True


def import_reference(src: str):
    sys.path.insert(0, src)
    from neurobrau_cli import settings, storage  # noqa: PLC0415
    from neurobrau_cli.conversation import Message  # noqa: PLC0415
    from neurobrau_cli.providers.base import TokenUsage  # noqa: PLC0415
    from neurobrau_cli.usage import UsageRecord  # noqa: PLC0415

    return storage, settings, Message, TokenUsage, UsageRecord


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


def generate(src: str, out: Path) -> None:
    storage, settings, Message, TokenUsage, UsageRecord = import_reference(src)
    conversations = [
        ("ollama_unicode", storage.StoredConversation(
            id="0b6f8e52-2c1d-4c7e-9a3b-5d2e1f0a9c41",
            created_at=at("2026-10-21T12:00:00.000+00:00"),
            updated_at=at("2026-10-21T12:03:41.512+00:00"),
            provider="ollama", model="qwen3.5:4b",
            messages=(
                Message("user", "Grüße, 世界! Ünïcödé 🎉\nLine two\twith tab"),
                Message("assistant", 'Quotes " and backslash \\ and <tag> & more'),
                Message("user", ""),
                Message("assistant", "Answer after an empty prompt."),
            ),
            requests=(
                UsageRecord(at("2026-10-21T12:01:00.000+00:00"), "ollama", "qwen3.5:4b",
                            TokenUsage(input_tokens=12, output_tokens=5, total_tokens=17)),
                UsageRecord(at("2026-10-21T12:03:41.000+00:00"), "ollama", "qwen3.5:4b",
                            TokenUsage(input_tokens=30, output_tokens=0), status="incomplete"),
            ),
        )),
        ("openai_usage", storage.StoredConversation(
            id="7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f",
            created_at=at("2026-10-22T05:54:00.123+00:00"),
            updated_at=at("2026-10-22T05:54:29.609+00:00"),
            provider="openai", model="gpt-6-luna",
            messages=(Message("user", "What is a neural network?"),
                      Message("assistant", "A model made of layers.")),
            requests=(
                UsageRecord(at("2026-10-22T05:54:29.609+00:00"), "openai", "gpt-6-luna",
                            TokenUsage(1000, 200, 1200, 600, 100, 50), "default"),
            ),
        )),
        ("legacy_no_provider", storage.StoredConversation(
            id="3f2a1b0c-8d7e-4f6a-9b5c-4d3e2f1a0b9c",
            created_at=at("2026-09-01T08:00:00.000+00:00"),
            updated_at=at("2026-09-01T08:05:00.000+00:00"),
            messages=(Message("user", "Hello"), Message("assistant", "Hi!")),
        )),
        ("empty_after_reset", storage.StoredConversation(
            id="a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
            created_at=at("2026-10-22T09:00:00.000+00:00"),
            updated_at=at("2026-10-22T09:00:00.000+00:00"),
            provider="openai", model="gpt-5",
        )),
    ]
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        store = storage.HistoryStore(Path(tmp) / "neurobrau" / "conversations")
        for name, conversation in conversations:
            store.save(conversation)
            shutil.copyfile(store.path_for(conversation.id), out / f"conversation_{name}.json")
        store.save_active("ollama", "qwen3.5:4b")
        shutil.copyfile(store.active_path, out / "active.json")

        prefs = settings.UserSettings(answer_style="brief")
        prefs = prefs.with_model("ollama", "qwen3.5:4b", reasoning="on", max_output_tokens=600,
                                 max_output_tokens_set=True)
        prefs = prefs.with_model("ollama", "hf.co/org/model:Q4_K_M", reasoning="high")
        prefs = prefs.with_model("openai", "gpt-6-luna", max_output_tokens=None,
                                 max_output_tokens_set=True)
        prefs = prefs.with_model("openai", "gpt-5", reasoning="default")
        settings_path = Path(tmp) / "config" / "neurobrau" / "settings.json"
        warning = settings.SettingsStore(settings_path).save(prefs)
        if warning:
            raise SystemExit(warning)
        shutil.copyfile(settings_path, out / "settings.json")


def check(src: str, directory: Path) -> None:
    """Python's view of Go-written files: what Python would write back."""
    storage, settings, *_ = import_reference(src)
    result: dict[str, object] = {"conversations": {}, "warnings": []}
    store = storage.HistoryStore(directory / "neurobrau" / "conversations")
    for path in store.list_paths():
        try:
            conversation = store.load(path)
        except storage.HistoryError as error:
            result["warnings"].append(str(error))
            continue
        text = json.dumps(storage.to_json(conversation), ensure_ascii=False, indent=2) + "\n"
        result["conversations"][path.name] = text
    active, warning = store.load_active()
    result["active"] = list(active) if active else None
    if warning:
        result["warnings"].append(warning)
    settings_store = settings.SettingsStore(directory / "config" / "neurobrau" / "settings.json")
    loaded, warnings = settings_store.load()
    result["warnings"].extend(warnings)
    result["settings"] = json.dumps(settings.to_json(loaded), ensure_ascii=False, indent=2) + "\n"
    result["settings_writable"] = settings_store.writable
    json.dump(result, sys.stdout, ensure_ascii=False)


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("generate", "check"):
        raise SystemExit(__doc__)
    command, src, directory = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    (generate if command == "generate" else check)(src, directory)


if __name__ == "__main__":
    main()
