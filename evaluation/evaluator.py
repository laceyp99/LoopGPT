import asyncio
import json
from mido import MidiFile
from datetime import datetime
from rich.live import Live
from rich.table import Table
import logging
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import gemini_api, claude_api, o_api, gpt_api
from src.midi_processing import loop_to_midi
from evaluation import tests

logging.disable(logging.INFO)

# Load model list and rate limiting details from a JSON file
with open('model_list.json', 'r') as f:
    model_info = json.load(f)

LOG_FILE = "evaluation_log.json"

def append_log(entry):
    """Append an entry to the JSON log file, keeping it as a list."""
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Log file is not a JSON list")
            except json.JSONDecodeError:
                data = []
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    else:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([entry], f, indent=2)

def save_generation_messages(provider, model, use_thinking, root, scale, duration, messages):
    # Convert bool to readable folder name
    thinking_folder = "thinking" if use_thinking else "no-thinking"
    
    # Create directory path
    base_dir = os.path.join(
        "Generations",
        provider,
        model,
        thinking_folder
    )
    os.makedirs(base_dir, exist_ok=True)

    # File-safe naming
    safe_name = f"{root}_{scale}_{duration}.json"

    # Save file
    file_path = os.path.join(base_dir, safe_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)

async def call_model_async(prompt, model_choice, temp, use_thinking, translate_prompt_choice):
    """
    Async wrapper for calling the appropriate model API to generate a loop or prompt.
    """
    def sync_call():
        """
        Calls the appropriate model API to generate a loop or prompt.

        Args:
            prompt (str): The input prompt for the model.
            model_choice (str): The selected model.
            temp (float): The sampling temperature.
            use_thinking (bool): Whether extended thinking is enabled.
            translate_prompt_choice (bool): Whether to translate the prompt.

        Returns:
            tuple: Generated loop, messages, and cost.
        """
        pt_cost = 0
        start_time = time.time()
        if model_choice in model_info["models"]["OpenAI"]:
            if model_info["models"]["OpenAI"][model_choice]["extended_thinking"]:
                if translate_prompt_choice:
                    prompt_translated, messages, pt_cost = o_api.prompt_gen(prompt, model_choice)
                    loop, messages, loop_cost = o_api.loop_gen(prompt_translated, model_choice)
                else:
                    loop, messages, loop_cost = o_api.loop_gen(prompt, model_choice)
            else:
                if translate_prompt_choice:
                    prompt_translated, messages, pt_cost = gpt_api.prompt_gen(prompt, model_choice, temp)
                    loop, messages, loop_cost = gpt_api.loop_gen(prompt_translated, model_choice, temp)
                else:
                    loop, messages, loop_cost = gpt_api.loop_gen(prompt, model_choice, temp)

        elif model_choice in model_info["models"]["Google"]:
            if translate_prompt_choice:
                prompt_translated, messages, pt_cost = gemini_api.prompt_gen(prompt, model_choice, temp, use_thinking)
                loop, messages, loop_cost = gemini_api.loop_gen(prompt_translated, model_choice, temp, use_thinking)
            else:
                loop, messages, loop_cost = gemini_api.loop_gen(prompt, model_choice, temp, use_thinking)

        elif model_choice in model_info["models"]["Anthropic"]:
            if translate_prompt_choice:
                prompt_translated, messages, pt_cost = claude_api.prompt_gen(prompt, model_choice, temp, use_thinking)
                loop, messages, loop_cost = claude_api.loop_gen(prompt_translated, model_choice, temp, use_thinking)
            else:
                loop, messages, loop_cost = claude_api.loop_gen(prompt, model_choice, temp, use_thinking)

        else:
            raise ValueError("Invalid Model Selected")

        time_elapsed = time.time() - start_time
        total_cost = pt_cost + loop_cost if translate_prompt_choice else loop_cost
        return loop, messages, total_cost, time_elapsed

    # Run sync call in separate thread to allow async concurrency
    return await asyncio.to_thread(sync_call)

def run_midi_tests(midi_data, root, scale, duration):
    four_bars = tests.four_bars(midi_data)
    key_test = tests.scale_test(midi_data, root, scale)
    duration_test = tests.duration_test(midi_data, duration)
    return {
        "bar_count_pass": four_bars,
        "in_key_pass": key_test,
        "note_length_pass": duration_test,
        "output_pass": four_bars and key_test and duration_test,
    }

async def evaluate_model(provider, model, prompt, semaphores, results, use_thinking, root, scale, duration):
    async with semaphores[provider]:
        
        midi_data, messages, cost, time_elapsed = await call_model_async(
            prompt=prompt,
            model_choice=model,
            temp=0.3,
            use_thinking=use_thinking,
            translate_prompt_choice=False
        )        
        midi_file = MidiFile()
        if provider == "Google":
            loop_to_midi(midi_file, midi_data, times_as_string=True)
        else:
            loop_to_midi(midi_file, midi_data, times_as_string=False)
        test_results = run_midi_tests(midi_file, root, scale, duration)

        result = {
            # "timestamp": datetime.datetime.now(datetime.UTC),
            "provider": provider,
            "model": model,
            "use_thinking": use_thinking,
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
        append_log(result)
        save_generation_messages(provider, model, use_thinking, root, scale, duration, messages)


async def main():
    roots = ["C", "A", "G"]
    scales = ["Major", "Minor"]
    durations = ["quarter", "eighth"]

    prompts = [
        (f"an arpeggiator in {root} {scale} using only {duration} note lengths", root, scale, duration)
        for root in roots
        for scale in scales
        for duration in durations
    ]

    # Build provider → models dict from JSON
    models_by_provider = {
        "OpenAI": list(model_info["models"]["OpenAI"].keys()),
        "Google": list(model_info["models"]["Google"].keys()),
        "Anthropic": list(model_info["models"]["Anthropic"].keys())
    }

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

    results = []

    table = Table(title="Model Evaluation Progress")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Bar Count")
    table.add_column("In Key")
    table.add_column("Note Length")
    table.add_column("Pass/Fail")

    async def runner():
        tasks = []
        for prompt, root, scale, duration in prompts:
            for provider, models in models_by_provider.items():
                for model in models:
                    # Always test without thinking
                    tasks.append(evaluate_model(provider, model, prompt, semaphores, results, use_thinking=False, root=root, scale=scale, duration=duration))

                    # If extended_thinking is supported, also test with thinking
                    ext_think = model_info["models"][provider][model]["extended_thinking"]
                    if ext_think:
                        tasks.append(evaluate_model(provider, model, prompt, semaphores, results, use_thinking=True, root=root, scale=scale, duration=duration))

        await asyncio.gather(*tasks)

    with Live(table, refresh_per_second=4) as live:
        task = asyncio.create_task(runner())
        while not task.done():

            table = Table(title="Model Evaluation Progress")
            table.add_column("Provider")
            table.add_column("Model")
            table.add_column("Prompt")
            table.add_column("Bar Count")
            table.add_column("In Key")
            table.add_column("Note Length")
            table.add_column("Pass/Fail")

            for r in results:
                table.add_row(
                    r["provider"],
                    r["model"],
                    r["prompt"]["full_prompt"],
                    "Yes" if r["use_thinking"] else "No",
                    "✅" if r["bar_count_pass"] else "❌",
                    "✅" if r["in_key_pass"] else "❌",
                    "✅" if r["note_length_pass"] else "❌",
                    "✅" if r["output_pass"] else "❌"
                )
            live.update(table)
            await asyncio.sleep(0.1)
        await task

if __name__ == "__main__":
    asyncio.run(main())
