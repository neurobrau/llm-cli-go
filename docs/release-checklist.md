# Release checklist — NeuroBräu Go 0.4.0

`0.4.0-dev` is the release candidate of the 0.4 milestone. Nothing here is
automated: the owner runs the checks and performs every release action.

## Prerequisites

- Go 1.26 or newer, Linux or macOS (the PTY tests run on Linux only).
- A clean, reviewed working tree: `git status --short` shows only the
  intended changes (and no stray files staged).
- Optional: the Python 0.3.0 reference checked out next to this repository
  (`../llm-cli-python`) with its virtual environment (`.venv`), for the
  parity checks.
- Optional: a local Ollama server with at least one small local model, for
  the manual smoke test. No OpenAI key is needed for any automated check.

## 1. Offline verification (no network, credentials or user files)

```bash
gofmt -l .                       # must print nothing
go vet ./...
go test -count=1 ./...
go test -race -count=1 ./...
go test -race -count=10 -run OnPTY ./cmd/neurobrau   # terminal restoration, repeated
go build -o bin/neurobrau ./cmd/neurobrau
CGO_ENABLED=0 go build -o bin/neurobrau-static ./cmd/neurobrau
./bin/neurobrau --version        # NeuroBräu Go CLI 0.4.0-dev
git diff --check
git diff --cached --check
git status --short
```

All tests use simulated screens, pseudo-terminals, fake providers,
`httptest` servers and temporary HOME/XDG directories.

## 2. Optional Python parity

```bash
NEUROBRAU_PYTHON_SRC=$PWD/../llm-cli-python/src \
NEUROBRAU_PYTHON=$PWD/../llm-cli-python/.venv/bin/python \
  go test -count=1 -v -run Python ./... | grep -E '^(---|ok)|cases compared'
git -C ../llm-cli-python status --short   # must stay empty
```

Expect every `TestPython…` test to pass (pricing: about 13,800 cases;
settings resolution: 23,328 cases, 0 differ). See [`parity.md`](parity.md)
for the documented differences.

## 3. Manual smoke test (owner only)

Use throwaway directories so real history and settings are never touched:

```bash
d=$(mktemp -d)
export HOME=$d XDG_STATE_HOME=$d/state XDG_CONFIG_HOME=$d/config

# Local Ollama (no API cost, nothing is pulled):
./bin/neurobrau --provider ollama "Say hello in five words."
./bin/neurobrau --provider ollama --stream "Count to five."
./bin/neurobrau --provider ollama --plain          # /usage, /reset, /exit
./bin/neurobrau --provider ollama --stream          # fullscreen: F2-F6, /usage, Ctrl+C
./bin/neurobrau --provider ollama --no-history      # nothing under $d/state afterwards
cat "$d/config/neurobrau/settings.json"; ls -la "$d/state/neurobrau/conversations"

# Terminal restoration: in another shell, `kill -TERM <pid>` of a running
# fullscreen session must return a usable prompt (exit status 143).
```

OpenAI (makes small **paid** requests; only if the owner chooses to): export
`OPENAI_API_KEY` in that shell only, run one one-shot prompt and one
fullscreen exchange, and compare `/usage`'s **estimated** cost with the
OpenAI usage dashboard. The estimate uses the pricing catalog verified on
**2026-10-22**; it is not a billed amount, and the invoice is authoritative.
Remove the temporary directory afterwards (`rm -rf "$d"`).

## 4. Release (owner only, after approval)

1. Set `Version` in `internal/cli/cli.go` to `0.4.0` (and the README /
   test expectations that show it), rerun section 1.
2. Commit with a message of your choice; review `git show --stat`.
3. Tag and push: `git tag -a v0.4.0 -m "NeuroBräu Go 0.4.0"`,
   `git push origin main v0.4.0`.
4. Optionally create a GitHub release from the tag with notes based on the
   README status section and [`parity.md`](parity.md), attaching binaries
   built with `CGO_ENABLED=0`.
5. Afterwards set the version to the next development value (for example
   `0.5.0-dev`).

Pricing catalog refreshes, new providers and the TypeScript edition are
separate work.
