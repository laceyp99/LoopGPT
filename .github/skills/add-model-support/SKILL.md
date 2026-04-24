---
name: add-model-support
description: 'Add support for a newly announced model from an existing provider. Use when updating model_list.json, checking provider pricing and rate limits, verifying src/*_api.py compatibility, and adjusting app.py UI controls for temperature, thinking, or effort support.'
argument-hint: 'Provider name, model name, and official announcement or API docs URL if available'
---

# Add Model Support

Use this skill when a provider already supported by LoopGPT announces a new model and you need to wire it into the repo safely.

This workflow is for existing providers only. Do not use it to add a brand-new provider module or broad evaluation coverage.

## Inputs

- Provider name.
- Model identifier.
- Official source URLs when available.

If the user does not provide a URL, search the web first and prefer official release notes, pricing pages, model docs, and API references.

## What This Skill Produces

- A researched update to `model_list.json` for the new model.
- Any minimum compatibility changes required in the matching provider module under `src/`.
- Any minimum UI changes required in `app.py` so the model appears and unsupported controls stay hidden.
- A summary of sources, assumptions, touched files, and validation results.

## Procedure

1. Confirm scope before editing.
   - Only proceed if the provider already exists in this repo.
   - If the request actually requires a new provider, stop and ask for a broader workflow.

2. Research the model from official sources.
   - Prefer vendor docs over third-party summaries.
   - Capture the public model identifier, pricing, context or max token limits, published rate limits, and any request-parameter constraints relevant to this repo.
   - Specifically determine whether the model supports or restricts temperature, extended thinking, or effort-style reasoning controls.
   - If official data is incomplete, do not invent values. Record the gap and ask for maintainer direction if the missing field blocks a safe edit.

3. Update the model registry.
   - Edit `model_list.json` under the existing provider key.
   - Preserve the current schema and nearby provider conventions.
   - Add `extended_thinking`, `effort_options` when applicable, `max_tokens`, `cost`, and `rate_limits` only from supported evidence.

4. Verify the provider module.
   - Inspect the matching file in `src/` and check whether the new model works with the current request construction, parameter names, parsing path, and cost calculation.
   - Make the smallest provider-side change needed.
   - Keep changes local to the provider unless a real compatibility constraint forces a nearby adjustment.

5. Verify the UI behavior.
   - Inspect `app.py` for provider and model dropdown behavior plus conditional controls.
   - Confirm the new model appears through the existing provider list.
   - Check whether the model should hide temperature, show thinking, or expose effort options.
   - Update hard-coded exceptions only when the new model truly requires them.

6. Validate immediately after the first substantive edit.
   - Prefer a focused syntax, import, or error check for the touched files.
   - If there is no narrow executable check, use the most local validation available and report what remains manual.
   - Do not run the long evaluation scripts for this workflow.

7. Report the outcome.
   - Summarize the official sources used.
   - List the fields added or changed in `model_list.json`.
   - Call out any provider-module or UI logic changes.
   - State assumptions, missing vendor details, and validation status.

## Project Notes

Use the repo-specific guide at [repo integration notes](./references/repo-integration.md) for the current file map, schema expectations, and app/provider control points.

## Completion Criteria

- The model is present under the correct provider in `model_list.json`.
- The matching provider module still uses valid request parameters for that model.
- `app.py` shows the model and only exposes controls the model supports.
- The final response includes sources, assumptions, touched files, and validation results.