# Repo Integration Notes

Use these notes to keep edits aligned with the current LoopGPT implementation.

## File Map

- `model_list.json`: canonical model registry and UI capability source for cloud providers.
- `app.py`: provider list, model dropdown, and conditional UI visibility for temperature, thinking, and effort controls.
- `src/openai_api.py`: OpenAI-family request construction and token pricing logic.
- `src/claude_api.py`: Anthropic request construction and thinking behavior.
- `src/gemini_api.py`: Google request construction and capability handling.
- `src/ollama_api.py`: local-model special case; usually reference-only for this workflow.
- `src/runs.py`: central routing. For this skill, treat it as reference-only unless an existing-provider model addition exposes a real routing assumption.
- `AGENTS.md`: repo guidance for model and provider support work.

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

## app.py Control Points

These functions control the model-selection experience and should be checked whenever a new model is added:

- `get_providers()`
- `get_models_for_provider(provider)`
- `update_model_choices(provider)`
- `update_temp_visibility(model_choice, use_thinking)`
- `update_thinking_visibility(model_choice)`
- `update_effort_visibility(model_choice)`

Important current behavior:

- Provider choices come from `model_info["models"]`, with Ollama appended dynamically.
- Temperature can be hidden for reasoning models and for specific hard-coded model exceptions.
- Thinking visibility is gated by `extended_thinking` plus a few explicit exclusions.
- Effort visibility is driven by the presence of `effort_options` in provider metadata.

When adding a model, first prefer metadata-driven behavior. Only add a hard-coded exception if the model really breaks the existing assumptions.

## Provider Module Checks

For the matching provider file in `src/`, verify all of the following before deciding no code change is needed:

- The model identifier is accepted by the provider API call path.
- The module uses the right parameter names for temperature, thinking, or effort controls.
- The cost calculation matches the registry fields used by that provider.
- Structured output parsing still works for loop generation.

Keep changes minimal. This workflow is not for creating a new provider module.

## Validation Guidance

- Run the narrowest error or syntax check available for touched files.
- If UI logic changes, confirm the new model appears in the provider dropdown and the control visibility matches the researched capabilities.
- Do not run the large evaluation scripts for this task.

## Reporting Expectations

The final report should include:

- Official sources used.
- Metadata fields added or changed.
- Provider-module and UI changes, if any.
- Any gaps in published pricing, rate limits, or parameter support.
- What was validated automatically and what still needs manual confirmation.