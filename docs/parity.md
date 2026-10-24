# Parity with Python 0.3.0

NeuroBräu Go `0.4.0-dev` implements the behavior of the Python 0.3.0
reference ([`../llm-cli-python`](https://github.com/neurobrau/llm-cli-python))
as specified in [`tui-spec.md`](tui-spec.md). Shared files (conversations,
`active.json`, `settings.json`) are interchangeable in both directions.

This page lists every known difference, classified as:

- **Fixed** — the Go edition now behaves like Python (or the spec).
- **Intentional** — Go differs on purpose; files and observable contracts
  stay compatible.
- **Deferred** — a known gap, with the reason it is left open.

Storage-level details (accepted JSON edge cases) are listed in
[`storage.md`](storage.md#known-differences).

## How parity is checked

Optional checks run the read-only Python sources on synthetic data in temp
directories (no network, no real user files, bytecode writing disabled):

```bash
NEUROBRAU_PYTHON_SRC=$PWD/../llm-cli-python/src \
NEUROBRAU_PYTHON=$PWD/../llm-cli-python/.venv/bin/python \
  go test -count=1 -run Python ./...
```

| Check | Compares |
|---|---|
| `TestPythonConversationFixtures`, `TestPythonActiveAndSettingsFixtures` (storage) | Go reads Python-written files and writes them back byte for byte |
| `TestPythonReadsGoFiles` (storage), `TestPythonReadsContinuedConversation` (session), `TestPythonReadsStreamedConversation` (chat) | Python reads Go-written files without warnings and writes them back byte for byte |
| `TestPythonPricingParity` (chat) | about 13,800 usage-record cases: per-request status, exact amount and notes, panel cost, cost line and the whole `/usage` text |
| `TestPythonSettingsParity` (chat) | 23,328 combinations of reasoning support, saved preferences, command-line values and strictness: effective reasoning, max output tokens, notices and errors |
| `TestWrapParagraphsMatchesPython` (tui) | conversation text wrapping |

## Fixed in 0.4.0-dev

| Area | Before | Now |
|---|---|---|
| `"service_tier": ""` in a usage record | read as "not reported" (priced as Standard, approximately) and dropped when written back | kept as reported but empty: priced as `unavailable` (`no pricing for the  tier of …`, as in Python) and written back; absent and `null` still assume Standard |
| `/usage` for a hand-edited record without any counts | printed `Token usage: not reported` | no count lines, as in Python; the cost line says `unavailable (token usage was not reported)` |
| `--reasoning-effort off` on a model without reasoning | exit 2 (`… does not support reasoning; 'off' is not available.`) | accepted like `none`: no parameter is sent |
| Interactive start when the reasoning lookup fails and nothing is chosen | no notice | `Could not check reasoning support for <model>: …` notice, as in Python (the lookup is cached; it is Ollama's `/api/show`) |
| SIGTERM / SIGHUP in the fullscreen UI | the process died with the terminal left in fullscreen raw mode (Python too) | the request is cancelled, the terminal restored, exit code 143 / 129 (see below) |
| Version | `0.1.0-dev` | `0.4.0-dev` everywhere (`--version`, banners, TUI title, help) |

## Intentional differences (compatible)

**Responsiveness and safety**

- Network work in the fullscreen UI (requests, provider/model checks,
  model discovery) runs in the background; the screen keeps redrawing,
  and Ctrl+C cancels it. Python blocks the UI.
- Cancelled reads stop at once, so the spec's "still closing" window (up
  to 1 s) rarely applies.
- SIGTERM/SIGHUP restore the terminal in the fullscreen UI (Python does
  not handle them). The plain chat and one-shot prompts keep the default
  signal behavior; they leave no terminal state to restore.
- API keys that appear in provider error text are redacted.
- A broken stdout pipe ends the process by SIGPIPE (exit status 141 in a
  shell) instead of an explicit `exit(141)`.

**Conversations and usage**

- A non-streaming answer cut off early (token limit) is shown with a
  warning and recorded as an `incomplete` usage record, but not saved as a
  message (as for streamed answers in both editions). Python 0.3.0 saves it.
- A cancelled request with no text shown prints `Request cancelled.`;
  Python always adds "The partial response was not saved."
- Streamed `<think>` blocks are passed on as reasoning while they arrive
  (never shown as answer text); Python holds the whole block.
- `/reset` always starts an empty conversation, as the spec says; Python
  reuses a conversation that has usage records but no messages.
- `--model` without `--provider` means OpenAI (spec); Python matches the
  model under any provider.
- Conversation files of an unknown provider are skipped; Python adopts them
  and then falls back to defaults.
- Without a home directory both history and settings are disabled; Python
  decides each separately.
- One-shot prompts use flags and built-in defaults only, never saved
  settings (task requirement); Python applies saved settings to one-shot
  prompts.

**Providers and errors**

- Failed, refused or empty OpenAI responses exit 1; a streamed answer cut
  off exits 8. Python's one-shot prints the (possibly empty) text and exits
  0.
- HTTP 403 maps to the authentication exit code 4 (Python: 1).
- Ollama `model-default` on a model with named thinking levels sends no
  `think` field (spec §13); Python sends `think: false`.
- A rejected default reasoning effort is retried without the parameter in
  one-shot prompts too (spec §12); Python only retries in the chat.
- A one-shot Ollama prompt whose model cannot be looked up fails at once
  (one error line); Python prints a note first and then fails.
- The one-shot prompt is sent as a one-item message list (same meaning).
- Undecodable OpenAI replies say `Invalid response from OpenAI.`.

**Command line and screen**

- Integer flags use Go parsing (`1_000`, ` 5` are rejected); invalid-choice
  errors quote with `"…"`; single-dash long flags (`-usage`) are accepted;
  `--version` with a prompt exits 2.
- Picking the value that is already saved in F4–F6 writes nothing.
- A failed settings save also shows `WARNING | Settings not saved`; an
  unknown settings location is a warning instead of silence.
- Menus are as wide as their title/footer + 7 (the spec's + 6 cuts the
  closing `]`).
- Large token counts abbreviate cleanly (`999,999` → `1M`; Python shows
  `1e+03k`); control characters in answers are drawn as `?`.
- The banner and title name the Go CLI; small help-text wording differs.
- `--usage` applies to one-shot and plain modes; the fullscreen UI shows
  usage in its panel and with `/usage`.

## Deferred

| Gap | Reason |
|---|---|
| Wide characters (CJK, emoji) count as one column | accepted by the spec; needs width-aware layout throughout |
| No in-app line editing or input history in the plain chat | out of scope for Phase 1 (terminal line discipline only) |
| SIGKILL, power loss or a crashed terminal emulator leave the terminal in raw mode | cannot be caught; run `reset` (or `stty sane`) in the shell |
| On a real hangup the process may exit with 1 instead of 129 | if the closed terminal is noticed before SIGHUP is delivered; nothing is saved either way |
| Pricing catalog fixed at 2026-10-22 | a catalog refresh is separate, owner-approved work |
| A provider that reports `service_tier: ""` in a live response is treated as not reported | the SDKs cannot tell empty from absent; no provider is known to send it |
| Prices exceeding Python `Decimal`'s 28 significant digits | Go's arithmetic is exact at any size; Python would round — not reachable with real token counts |
