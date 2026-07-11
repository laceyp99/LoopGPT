---
name: add-model-support
description: Add support for a newly announced model from an existing Conductor Core provider. Use when researching official model identifiers, pricing, limits, and parameter support; updating the packaged model registry; verifying provider request and response compatibility; and checking downstream Conductor Main or Eval capability behavior.
---

# Add Model Support

Use this skill when a provider already supported by Conductor Core announces a new model and you need to wire it into the package safely.

This workflow is for existing providers only. Do not use it to add a brand-new provider module or broad evaluation coverage.

## Inputs

- Provider name.
- Model identifier.
- Official source URLs when available.

If the user does not provide a URL, search the web first and prefer official release notes, pricing pages, model docs, and API references.

## What This Skill Produces

- A researched update to the packaged `model_list.json` for the new model.
- Any minimum compatibility changes required in `conductor_core.providers`.
- Any minimum downstream changes required for Conductor Main controls or Eval model selection.
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
   - Edit `src/conductor_core/resources/model_list.json` under the existing provider key.
   - Preserve the current schema and nearby provider conventions.
   - Add `extended_thinking`, `effort_options` when applicable, `max_tokens`, `cost`, and `rate_limits` only from supported evidence.

4. Verify the provider module.
   - Inspect the matching file under `src/conductor_core/providers/` and check whether the new model works with the current request construction, parameter names, parsing path, and cost calculation.
   - Make the smallest provider-side change needed.
   - Keep changes local to the provider unless a real compatibility constraint forces a nearby adjustment.

5. Verify downstream consumers.
   - When working in the suite monorepo, inspect `apps/conductor-main/src/conductor_main/app.py` for provider and model dropdown behavior plus conditional controls.
   - Confirm the new model appears through the existing provider list.
   - Check whether the model should hide temperature, show thinking, or expose effort options.
   - Update hard-coded exceptions only when the new model truly requires them.
   - Inspect `projects/conductor-eval` only if model grouping, reasoning variations, or rate-limit behavior needs a compatibility change.
   - If Core is checked out as a standalone repo, report downstream checks as follow-up work instead of assuming sibling repositories exist.

6. Validate immediately after the first substantive edit.
   - Prefer a focused syntax, import, or error check for the touched files.
   - If there is no narrow executable check, use the most local validation available and report what remains manual.
   - Do not run the long evaluation scripts for this workflow.

7. Report the outcome.
   - Summarize the official sources used.
   - List the fields added or changed in the packaged `model_list.json`.
   - Call out any provider-module or UI logic changes.
   - State assumptions, missing vendor details, and validation status.

## Project Notes

Use the repo-specific guide at [repo integration notes](./references/repo-integration.md) for the current file map, schema expectations, and app/provider control points.

## Completion Criteria

- The model is present under the correct provider in `src/conductor_core/resources/model_list.json`.
- The matching provider module still uses valid request parameters for that model.
- Available downstream clients expose only controls the model supports.
- The final response includes sources, assumptions, touched files, and validation results.
