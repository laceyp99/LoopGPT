# LoopGPT — Retired

> [!IMPORTANT]
> **This repository is retired and archived.** Active development has moved to
> the [Conductor project suite](https://github.com/laceyp99/conductor-main/tree/main). No new issues, pull requests, or releases will
> be accepted here.

LoopGPT began as a single application that generated 4-bar MIDI loops from
natural-language prompts. It has been split into three independently versioned
projects, each with its own repository, documentation, tests, and CI:

| Project | Repository | Purpose |
|---|---|---|
| **Conductor Core** | [laceyp99/conductor-core](https://github.com/laceyp99/conductor-core) | Reusable prompt-to-MIDI generation engine: provider routing (OpenAI, Anthropic, Google, Ollama), loop models, MIDI conversion, playback helpers, and artifact persistence |
| **Conductor Main** | [laceyp99/conductor-main](https://github.com/laceyp99/conductor-main) | Interactive Gradio client: generation UI, piano-roll visualization, SoundFont playback, and generation history |
| **Conductor Eval** | [laceyp99/conductor-eval](https://github.com/laceyp99/conductor-eval) | Evaluation framework: model/prompt quality runs, deterministic MIDI checks, reports, and the analysis dashboard |

## Where to go

- **Want to run the app?** Start with [conductor-main](https://github.com/laceyp99/conductor-main).
- **Want the generation engine in your own script, service, or notebook?** Use [conductor-core](https://github.com/laceyp99/conductor-core).
- **Want to benchmark models and prompts?** Use [conductor-eval](https://github.com/laceyp99/conductor-eval).

## About this repository's history

The full monorepo history remains browsable here. The
[`pre-split`](https://github.com/laceyp99/LoopGPT/releases/tag/pre-split) tag
marks the last state before the extraction; each new repository carries the
filtered commit history of its own files. The final monorepo layout was:

```text
packages/conductor-core/   → now laceyp99/conductor-core
apps/conductor-main/       → now laceyp99/conductor-main
projects/conductor-eval/   → now laceyp99/conductor-eval
src/, evaluation/, app.py  → transitional compatibility wrappers (retired)
```

This repository is kept read-only for reference, provenance, and existing
links. It will not receive further updates.
