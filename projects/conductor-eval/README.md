# Conductor Evaluation Framework

A unified evaluation framework for testing MIDI loop generation across multiple AI models, with an interactive Plotly Dash dashboard for analyzing results.

This framework is not a replacement for its local pytest suite. Use pytest for
fast local checks of deterministic Eval code paths. Use the evaluator when you
want to measure prompt-to-model behavior, musical validity, latency, cost, or
reasoning-mode differences across real models.

## File Reference

| File | Description |
|------|-------------|
| `evaluator.py` | Main `Evaluator` class -- orchestrates generation, testing, and result saving |
| `checks.py` | MIDI validation functions (`scale_test`, `duration_test`) |
| `analysis.py` | Interactive Plotly Dash dashboard (up to 8 tabs, up to 22 charts, global filters, export) |


## Quick Start

### Run an Evaluation

```python
from conductor_eval import Evaluator

evaluator = Evaluator(temperature=0.0)

results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C", "G", "D"],
    models="openai",
    run_name="my_first_eval"
)
```

### Direct Script Safeguard

Running `py -3.12 -m conductor_eval.evaluator` directly is guarded because the
example in that file starts a broad cloud evaluation across multiple paid
providers. The script prints a warning and requires the exact confirmation
phrase `RUN CLOUD EVALUATION` before it creates an `Evaluator` or starts any
provider calls.

Pressing Enter, sending no input, or typing anything else aborts the script
without creating evaluation outputs. For smaller intentional runs, prefer the
Python API examples above so you can choose the prompts, roots, models, and run
name explicitly.

### Launch the Dashboard

```powershell
# Interactive run selection
.\.venv\Scripts\python.exe -m conductor_eval.analysis

# Direct path to a run
.\.venv\Scripts\python.exe -m conductor_eval.analysis evaluations/20260210_224954_arpeggiator_local_pt
```

The dashboard opens at `http://127.0.0.1:8050/`.

## Installation

From the `conductor-eval` project directory, create a virtual environment,
install a compatible published Core release with provider support, then install
Eval with its dashboard and development extras:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "conductor-core[providers]"
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard,dev]"
```

The explicit interpreter paths prevent `py -3.12 -m pip` from accidentally
installing into the registered global Python instead of this environment.

Key packages: `dash`, `dash-bootstrap-components`, `pandas`, `plotly`, `mido`, `rich`.

---

## Evaluator

### Basic Evaluation

```python
from conductor_eval import Evaluator

evaluator = Evaluator(temperature=0.0)

# Single prompt, multiple roots, one provider
results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C", "G"],
    models="openai",
    run_name="quarter_note_test"
)
```

The evaluator automatically appends `" in {root} {scale}"` to each prompt and runs both major and minor scales for every root.

### Multiple Prompts

Test different musical patterns in a single run:

```python
results = evaluator.evaluate(
    prompts=[
        "an arpeggiator using only quarter notes",
        "an arpeggiator using only eighth notes",
        "an arpeggiator using only sixteenth notes",
    ],
    roots=["C", "D", "E", "F", "G", "A", "B"],
    models="all",
    run_name="duration_comparison"
)
```

### Model Selection

The `models` parameter accepts several formats:

| Value | Description |
|-------|-------------|
| `"all"` | All models from all providers (cloud + Ollama) |
| `"openai"` | All OpenAI models |
| `"anthropic"` | All Anthropic models |
| `"google"` | All Google Gemini models |
| `"ollama"` | All local Ollama models |
| `["gpt-5", "claude-sonnet-4-6"]` | Specific models by name |

### Testing Reasoning Variations

When `test_reasoning=True`, the evaluator tests all thinking modes and effort levels for compatible models:

```python
results = evaluator.evaluate(
    prompts="complex chord progression",
    roots=["C", "G"],
    models=["o3", "claude-sonnet-4-5"],
    run_name="reasoning_test",
    test_reasoning=True
)
```

| Provider | Model Type | Variations |
|----------|------------|------------|
| OpenAI | `gpt-5.x` and `o`-series reasoning models | effort levels only; current families use `none` to `xhigh`, `minimal` to `high`, or `low` to `high` depending on model |
| Anthropic | Claude 4.x reasoning models | either effort levels only for Claude 5 models, `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, or `standard` / `w_reasoning` toggle for other reasoning-capable Claude 4.x models |
| Google | Gemini 3.x and 2.5 reasoning models | Gemini 3.x uses effort levels; Gemini 2.5 models use `standard` / `w_reasoning` toggle |
| Ollama | All | standard only in the evaluator |

### Configuring Tests

The `tests` parameter controls which validation tests run on generated MIDI:

```python
results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C"],
    models="openai",
    run_name="scale_only_test",
    tests=["scale"]  # Only run scale test, skip duration
)
```

| Test | Description | Auto-Detection |
|------|-------------|----------------|
| `scale` | Validates notes belong to the specified scale | Always uses root/scale from prompt |
| `duration` | Validates note durations match expected value | Detects from keywords: `quarter`, `eighth`, `sixteenth`, `16th`, `8th`, `half`, `whole` |

The `scale` test always runs since root and scale are always applied to prompts. Duration keywords are owned by `conductor_core.music` as `DURATION_KEYWORDS` and shared with the evaluator.

### Output Structure

Each evaluation run creates a timestamped directory:

```
evaluations/
└── 20260210_224954_my_first_eval/
    ├── config.json                    # Full evaluation configuration
    ├── summary.json                   # Aggregated results + statistics
    ├── core_artifacts/                # Core-owned MIDI, messages, and metadata
    ├── analysis/                      # Created by dashboard export
    │   └── dashboard.html
    └── results/
        └── OpenAI/
            └── gpt-5/
                └── an_arpeggiator_using_only_quarter_notes/
                    ├── C_major/
                    │   ├── loop.mid           # Generated MIDI file
                    │   ├── messages.json      # Chat history (for fine-tuning)
                    │   └── test_results.json  # Individual test results
                    └── C_minor/
                        └── ...
```

The evaluator intentionally retains `core_artifacts/` after copying MIDI and
messages into the report-oriented `results/` tree. Core owns generation
persistence, and retaining its canonical artifacts preserves provenance and
provider metadata for debugging. Eval does not selectively delete those files;
remove an entire completed run externally when its artifacts are no longer
needed.

When using `test_reasoning`, variation subfolders are created:

```
# With test_reasoning=True (effort levels as subfolders)
C_major/
├── none/                   # No reasoning effort
│   ├── loop.mid
│   ├── messages.json
│   └── test_results.json
├── low/
├── medium/
├── high/
└── xhigh/
```

#### config.json

Stores the full configuration used for the run:

```json
{
    "run_name": "my_first_eval",
    "timestamp": "20260207_143022",
    "prompts": ["an arpeggiator using only quarter notes"],
    "roots": ["C", "G"],
    "scales": ["major", "minor"],
    "models": [["OpenAI", "gpt-5"]],
    "tests": ["scale", "duration"],
    "test_reasoning": false,
    "temperature": 0.0
}
```

#### summary.json

Aggregated statistics for the entire run:

```json
{
    "run_id": "20260207_143022_my_first_eval",
    "totals": {
        "total_generations": 48,
        "successful_generations": 45,
        "failed_generations": 3,
        "overall_pass_count": 36,
        "overall_pass_rate": 0.75,
        "total_cost": 1.25,
        "total_time": 120.5
    },
    "by_model": {
        "gpt-5": {
            "provider": "OpenAI",
            "tested": 24,
            "passed": 20,
            "pass_rate": 0.833,
            "total_cost": 0.50,
            "avg_latency": 2.1
        }
    },
    "by_root": { "C": { "tested": 24, "passed": 18, "pass_rate": 0.75 } },
    "by_scale": { "major": { "tested": 24, "passed": 20, "pass_rate": 0.833 } }
}
```

#### test_results.json

Individual results for each generation:

```json
{
    "model": "gpt-5",
    "provider": "OpenAI",
    "prompt": "an arpeggiator using only quarter notes in C major",
    "original_prompt": "an arpeggiator using only quarter notes",
    "root": "C",
    "scale": "major",
    "config": {
        "use_thinking": false,
        "effort": null,
        "temperature": 0.0
    },
    "metrics": {
        "api_latency": 2.34,
        "cost": 0.0025
    },
    "tests": {
        "scale": {
            "ran": true,
            "params": { "root": "C", "scale": "major" },
            "total": 16,
            "correct": 16,
            "incorrect": 0,
            "pitches": { "correct": [0, 2, 4, 5, 7, 9, 11], "incorrect": [] }
        },
        "duration": {
            "ran": true,
            "params": { "duration": "quarter" },
            "detected_from_prompt": true,
            "total": 16,
            "correct": 16,
            "incorrect": 0,
            "lengths": {}
        },
        "overall_pass": true
    }
}
```


## Analysis
### Global Filters

A filter bar at the top of every page lets you narrow results by:

- **Models** -- Select which models to include
- **Root Notes** -- Filter by root note (e.g. C, F#, Eb)
- **Scale Type** -- Major, minor, or both
- **Variation** -- Standard and reasoning effort levels

All charts update in real time when filters change.

### Dashboard Tabs

#### Tab 1: Overview
- Metric cards: total generations, pass rate, best/worst model, total cost, average latency
- Overall pass rate by model (horizontal bar chart)

#### Tab 2: Model Performance
- Per-test breakdown: scale vs duration vs overall pass rate per model
- Major vs minor pass rate comparison per model
- Model x scale heatmap
- Model x root heatmap

#### Tab 3: Root & Scale
- Pass rate by root note
- Major vs minor pass rate per root note
- Full model x root+scale heatmap

#### Tab 4: Latency
- Latency distribution box plots per model
- Latency vs pass rate scatter plot

#### Tab 5: Cost
- Total cost by model
- Cost per generation vs pass rate scatter plot

#### Tab 6: Reasoning *(only shown when `test_reasoning=True`)*
- Effort impact delta: pass rate change across effort levels per model
- Reasoning toggle comparison: pass rate with thinking on vs off for toggle-based models
- Reasoning cost-effectiveness: cost vs pass rate scatter colored by effort level

#### Tab 7: Error Patterns
- Generation failure rate (API/conversion errors) per model
- Most common incorrect pitch classes by model (as note names)
- Incorrect intervals relative to prompted root per model (e.g. m3, P5 -- helps identify systematic confusions)
- Incorrect durations by model showing actual vs requested duration

### Exporting

Click the **Export Dashboard** button to save all charts as individual HTML files plus a combined `dashboard.html` to `evaluations/<run>/analysis/`. The number of exported charts depends on the run's features (16 base charts, plus 3 for reasoning when applicable):

```
evaluations/20260210_224954_arpeggiator_local_pt/
└── analysis/
    ├── dashboard.html              # Combined single-page dashboard
    ├── pass_rate_by_model.html
    ├── per_test_breakdown.html
    ├── incorrect_intervals.html
    ├── effort_impact_delta.html    # Only if test_reasoning was used
    └── ... (up to 19 chart files total)
```

The exported HTML files are self-contained and can be shared without a running server.

---

## Error Handling

The evaluator continues on failures, logging errors and saving partial results:

- API errors are captured in `test_results.json` with an `"error"` field
- Failed generations are counted in `summary.json` under `failed_generations`
- Core generation or MIDI conversion errors are logged but don't halt the evaluation
- All logs are written to `<output_dir>/run.log`

## Performance Notes

- **Cloud providers** run asynchronously with rate limiting based on RPM from `model_list.json`
- **Ollama** runs synchronously, sorted by model to minimize GPU memory swaps
- A live Rich progress table displays during evaluation with per-model pass rates, latency, and cost
- Large evaluations (many models x many prompts x many roots) can take significant time and incur API costs
