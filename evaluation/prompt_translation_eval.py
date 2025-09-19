from rich.live import Live
from rich.table import Table
from rich.console import Console
from mido import MidiFile
import logging
import asyncio
import time
import json
import sys
import os

# Imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import gemini_api, claude_api, o_api, gpt_api
import src.runs as runs
from src.midi_processing import loop_to_midi
from evaluation import sota_eval, tests

# Disable logging for cleaner output
logging.disable(logging.INFO)
console = Console(force_terminal=True)

# Load model list and rate limiting details from a JSON file
with open('model_list.json', 'r') as f:
    model_info = json.load(f)

def save_generation_messages(provider, model, translated, root, scale, duration, messages):
    """Save the generation messages to a structured directory for translation evaluation."""
    mode_folder = "translated" if translated else "no-translation"
    base_dir = os.path.join("Generations", provider, model, mode_folder)
    os.makedirs(base_dir, exist_ok=True)
    safe_name = f"{root}_{scale}_{duration}.json"
    with open(os.path.join(base_dir, safe_name), "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)

async def evaluate_model(provider, model, prompt, semaphores, results, translated, root, scale, duration):
    """Evaluate a specific model with or without prompt translation."""
    async with semaphores[provider]:
        # Call model with translation flag (use_thinking is always False here)
        try:
            midi_data, messages, cost, time_elapsed = await sota_eval.call_model_async(
                prompt=prompt,
                model_choice=model,
                temp=0.3,
                use_thinking=False,
                translate_prompt_choice=translated
            )
        except Exception as e:
            midi_data = None
            messages = [str(e)]
            cost = 0.0
            time_elapsed = 0.0

        try:
            # Convert generated loop to MIDI
            midi_file = MidiFile()
            if provider == "Google":
                loop_to_midi(midi_file, midi_data, times_as_string=True)
            else:
                loop_to_midi(midi_file, midi_data, times_as_string=False)
        except Exception as e:
            midi_file = MidiFile()
            messages.append(f"MIDI Conversion Error: {str(e)}")

        # Save MIDI
        os.makedirs(os.path.join("MIDI", model), exist_ok=True)
        safe_name = f"{root}_{scale}_{duration}{'_translated' if translated else '_standard'}.mid"
        midi_file.save(os.path.join("MIDI", model, safe_name))

        # Run tests
        test_results = tests.run_midi_tests(midi_file, root, scale, duration)
        result = {
            "provider": provider,
            "model": model,
            "translated": translated,
            "prompt": {
                "full_prompt": prompt,
                "root": root,
                "scale": scale,
                "duration": duration
            },
            "api_latency": time_elapsed,
            "cost": cost,
            **test_results
        }

        # Log and save
        results.append(result)
        sota_eval.append_log(result)
        save_generation_messages(provider, model, translated, root, scale, duration, messages)

models_by_provider = {
    "OpenAI": ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4o-2024-08-06"],
    "Google": ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "Anthropic": ["claude-3-5-haiku-20241022", "claude-3-haiku-20240307"]
}

async def main():
    # Prompt options
    roots = ["C", "A", "G"]
    scales = ["Major", "Minor"]
    durations = ["quarter", "eighth"]
    # Generate all combinations of prompts
    prompts = [
        (f"{root} {scale} arpeggiator using only {duration} notes", root, scale, duration)
        for root in roots
        for scale in scales
        for duration in durations
    ]
    total_prompt_count = len(prompts)
    
    # Expected total tests per (provider, model, translated)
    expected_tests = {}
    for provider, models in models_by_provider.items():
        for model in models:
            expected_tests[(provider, model, False)] = total_prompt_count   # no translation
            expected_tests[(provider, model, True)] = total_prompt_count    # with translation

    
    # Build semaphores dynamically from RPM
    semaphores = {}
    for provider, models in models_by_provider.items():
        rpms = []
        for model in models:
            rate_info = model_info["models"][provider][model]["rate_limits"]
            rpm = rate_info.get("RPM", 60)  # default 1 req/sec if not found
            rpms.append(rpm)
        # Set concurrency as RPM / 60 (rounded down, min 1)
        max_concurrent = max(1, min(rpms) // 60)
        semaphores[provider] = asyncio.Semaphore(max_concurrent)
    
    stats = {
        "no_translation": {"accuracy": [], "latency": [], "cost": []},
        "with_translation": {"accuracy": [], "latency": [], "cost": []}
    }
    
    results = []

    # Live table showing per-model aggregated progress
    table = Table(title="Prompt Translation Evaluation Progress")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Translated")
    table.add_column("Tested")
    table.add_column("Total")
    table.add_column("Pass Rate")
    table.add_column("Avg Latency (s)")
    table.add_column("Avg Cost")

    async def runner():
        tasks = []
        for prompt, root, scale, duration in prompts:
            for provider, models in models_by_provider.items():
                for model in models:
                    tasks.append(evaluate_model(provider, model, prompt, semaphores, results, translated=False, root=root, scale=scale, duration=duration))
                    tasks.append(evaluate_model(provider, model, prompt, semaphores, results, translated=True, root=root, scale=scale, duration=duration))
        await asyncio.gather(*tasks)

    with Live(table, console=console, refresh_per_second=4) as live:
        task = asyncio.create_task(runner())

        while not task.done():
            new_table = Table(title="Prompt Translation Evaluation Progress")
            new_table.add_column("Provider")
            new_table.add_column("Model")
            new_table.add_column("Translated")
            new_table.add_column("Tested")
            new_table.add_column("Total")
            new_table.add_column("Pass Rate")
            new_table.add_column("Avg Latency (s)")
            new_table.add_column("Avg Cost")

            # Aggregate results
            stats = {}
            for r in results:
                key = (r["provider"], r["model"], r["translated"])
                s = stats.setdefault(key, {"tested": 0, "passes": 0, "latency_sum": 0.0, "cost_sum": 0.0})
                s["tested"] += 1
                s["passes"] += 1 if r.get("output_pass") else 0
                s["latency_sum"] += r.get("api_latency", 0.0) or 0.0
                s["cost_sum"] += r.get("cost", 0.0) or 0.0

            for key, total in expected_tests.items():
                provider, model, translated = key
                s = stats.get(key, {"tested": 0, "passes": 0, "latency_sum": 0.0, "cost_sum": 0.0})
                tested = s["tested"]
                passes = s["passes"]
                avg_latency = (s["latency_sum"] / tested) if tested else 0.0
                avg_cost = (s["cost_sum"] / tested) if tested else 0.0
                pass_rate = (passes / tested * 100.0) if tested else 0.0

                new_table.add_row(
                    provider,
                    model,
                    "Yes" if translated else "No",
                    f"{tested}",
                    f"{total}",
                    f"{pass_rate:.1f}%",
                    f"{avg_latency:.2f}",
                    f"{avg_cost:.4f}"
                )

            live.update(new_table)
            await asyncio.sleep(0.25)

        await task
        live.update(new_table)

if __name__ == "__main__":
    asyncio.run(main())