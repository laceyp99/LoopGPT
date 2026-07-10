# Repo Integration Notes

Use these notes to keep edits aligned with Conductor Core and its current suite consumers.

## File Map

- `src/conductor_core/resources/model_list.json`: canonical cloud-model registry, pricing, limits, and capability source.
- `src/conductor_core/providers/openai.py`: OpenAI request construction, parsing, and cost calculation.
- `src/conductor_core/providers/anthropic.py`: Anthropic request construction, parsing, and reasoning behavior.
- `src/conductor_core/providers/google.py`: Google request construction, parsing, and reasoning behavior.
- `src/conductor_core/providers/ollama.py`: local-model discovery and request behavior; usually reference-only for announced cloud models.
- `src/conductor_core/routing.py`: central provider routing; change only when an existing-provider model exposes a real routing assumption.
- `tests/`: Core registry, provider, routing, and response-parsing coverage.
- `AGENTS.md`: repository guidance when the package is still inside the suite monorepo.

When sibling projects are available:

- `../../apps/conductor-main/src/conductor_main/app.py`: model dropdowns and metadata-driven temperature, thinking, and effort controls.
- `../../projects/conductor-eval/src/conductor_eval/evaluator.py`: evaluation model selection, rate limits, and reasoning variations.

## Model Registry Schema

The current schema is provider keyed under `models`.

Example shape:

```json
{
  "models": {
    "OpenAI": {
      "model-id": {
        "extended_thinking": true,
        "effort_options": ["low", "medium", "high"],
        "max_tokens": 128000,
        "cost": {
          "input": 1.25,
          "cached input": 0.125,
          "output": 10.0
        },
        "rate_limits": {
          "TPM": 2000000,
          "RPM": 10000,
          "TPD": 200000000
        }
      }
    }
  }
}
```

Keep field names and nesting consistent with nearby entries. If a provider does not publish one of these values, do not fabricate it.

## Conductor Main Control Points

These functions control the model-selection experience and should be checked whenever a new model is added:

- `get_providers()`
- `get_models_for_provider(provider)`
- `get_selected_model(provider, model_choice)`
- `get_model_settings(provider, model_choice, use_thinking=False)`
- `sync_model_capabilities(provider, model_choice, use_thinking=False)`

Important current behavior:

- Provider choices come from `model_info["models"]`, with Ollama appended dynamically.
- Temperature is hidden for supported reasoning configurations.
- Toggle-style thinking is driven by `extended_thinking` when no effort options exist.
- Effort visibility is driven by the presence of `effort_options` in provider metadata.

When adding a model, first prefer metadata-driven behavior. Only add a hard-coded exception if the model really breaks the existing assumptions.

## Provider Module Checks

For the matching provider file in `src/conductor_core/providers/`, verify all of the following before deciding no code change is needed:

- The model identifier is accepted by the provider API call path.
- The module uses the right parameter names for temperature, thinking, or effort controls.
- The cost calculation matches the registry fields used by that provider.
- Structured output parsing still works for loop generation.

Keep changes minimal. This workflow is not for creating a new provider module.

## Validation Guidance

- Run the narrowest error or syntax check available for touched files.
- Run focused Core registry, provider, routing, and response-parsing tests.
- If UI logic changes and Conductor Main is available, confirm the model appears in the provider dropdown and control visibility matches the researched capabilities.
- If reasoning or rate-limit metadata changes and Eval is available, run its focused model-selection tests.
- Do not run the large evaluation scripts for this task.

## Reporting Expectations

The final report should include:

- Official sources used.
- Metadata fields added or changed.
- Provider-module and UI changes, if any.
- Any gaps in published pricing, rate limits, or parameter support.
- What was validated automatically and what still needs manual confirmation.
