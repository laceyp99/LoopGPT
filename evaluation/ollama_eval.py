from rich.live import Live
from rich.table import Table
from rich.console import Console
from mido import MidiFile
import logging
import asyncio
import json
import time
import sys
import os

# Imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import ollama_api
import src.runs as runs
from src.midi_processing import loop_to_midi
from evaluation import tests, evaluator

# Disable logging for cleaner output
logging.disable(logging.INFO)
console = Console(force_terminal=True)

model_list = ollama_api.model_list
model_list.remove("phi4-reasoning:plus")
model_list.remove("qwen3:14b")

# Prompt options
roots = ["C", "A", "G"]
scales = ["Major", "Minor"]
durations = ["quarter", "eighth"]
# Generate all combinations of prompts
prompts = [
    (f"an arpeggiator in {root} {scale} using only {duration} note lengths", root, scale, duration)
    for root in roots
    for scale in scales
    for duration in durations
]
expected_total = len(prompts)
stats = {}
results = []

def build_table():
    table = Table(title="Ollama Model Evaluation Progress")
    table.add_column("Model")
    table.add_column("Tested")
    table.add_column("Total")
    table.add_column("Pass Rate")
    table.add_column("Avg Latency (s)")
    table.add_column("Avg Cost")
    for m in model_list:
        s = stats.get(m, {"tested": 0, "passes": 0, "latency_sum": 0.0, "cost_sum": 0.0})
        tested = s["tested"]
        passes = s["passes"]
        pass_rate = (passes / tested * 100.0) if tested else 0.0
        avg_latency = (s["latency_sum"] / tested) if tested else 0.0
        avg_cost = (s["cost_sum"] / tested) if tested else 0.0
        table.add_row(
            m,
            str(tested),
            str(expected_total),
            f"{pass_rate:.1f}%",
            f"{avg_latency:.2f}",
            f"{avg_cost:.4f}"
        )
    return table

with Live(build_table(), console=console, refresh_per_second=4) as live:
    for model in model_list:
        # print(model)
        for prompt, root, scale, duration in prompts:
            # print(prompt)
            start_time = time.time()
            loop, messages, cost = runs.generate_midi(
                model_choice=model,
                prompt=prompt,
                temp=0.3,
                translate_prompt_choice=False,
                use_thinking=False
            )
            time_elapsed = time.time() - start_time
            midi = MidiFile()
            loop_to_midi(midi, loop, times_as_string=False)
            safe_model_name = model.replace(":", "_size_").replace(".", "-")
            os.makedirs(os.path.join("MIDI", "Ollama", safe_model_name), exist_ok=True)
            midi.save(os.path.join("MIDI", "Ollama", safe_model_name, f"{prompt}.mid"))
            test_results = tests.run_midi_tests(midi, root, scale, duration)
            result = {
                "provider": "Ollama",
                "model": model,
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
            results.append(result)
            evaluator.append_log(result)
            evaluator.save_generation_messages("Ollama", safe_model_name, False, root, scale, duration, messages)
            mstats = stats.setdefault(model, {"tested": 0, "passes": 0, "latency_sum": 0.0, "cost_sum": 0.0})

            # Update stats and refresh table
            mstats["tested"] += 1
            mstats["passes"] += 1 if result.get("output_pass") else 0
            mstats["latency_sum"] += time_elapsed
            mstats["cost_sum"] += cost or 0.0
            live.update(build_table())