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
from src import gemini_api, claude_api, o_api, gpt_api
import src.runs as runs
from src.midi_processing import loop_to_midi
from evaluation import tests

# Disable logging for cleaner output
logging.disable(logging.INFO)
console = Console(force_terminal=True)

# Load model list and rate limiting details from a JSON file
with open('model_list.json', 'r') as f:
    model_info = json.load(f)

def append_log(entry):
    """ Append an entry to the JSON log file, keeping it as a list.
    
    Args:
        entry (dict): The log entry to append.
    
    Returns:
        None: The function writes directly to the log file.
    """
    # If the log file exists and has content
    if os.path.exists("evaluation_log.json") and os.path.getsize("evaluation_log.json") > 0:
        with open("evaluation_log.json", "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Log file is not a JSON list")
            except json.JSONDecodeError:
                data = []
            # Append the new entry to the existing list
            data.append(entry)
            f.seek(0)
            # Write the updated list back to the file
            json.dump(data, f, indent=2)
            f.truncate()
    # If the log file does not exist or is empty, create a new list with the entry
    else:
        with open("evaluation_log.json", "w", encoding="utf-8") as f:
            json.dump([entry], f, indent=2)

def save_generation_messages(provider, model, use_thinking, root, scale, duration, messages):
    """ Save the generation messages to a structured directory based on provider, model, and thinking state.

    Args:
        provider (str): The name of the provider (e.g., "OpenAI", "Google").
        model (str): The model name.
        use_thinking (bool): Whether extended thinking was used.
        root (str): The musical root note.
        scale (str): The musical scale.
        duration (str): The note duration.
        messages (list): The messages generated during the process.
    
    Returns:
        None: Saves the messages to a JSON file in a structured directory.
    """
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
            tuple: Generated loop, messages, cost, and time elapsed.
        """
        # Making the designated API call and measuring API latency
        start_time = time.time()
        loop, messages, total_cost = runs.generate_midi(
            model_choice=model_choice, 
            prompt=prompt, 
            temp=temp, 
            translate_prompt_choice=translate_prompt_choice, 
            use_thinking=use_thinking
        )
        time_elapsed = time.time() - start_time
        return loop, messages, total_cost, time_elapsed

    # Run sync call in separate thread to allow async concurrency
    return await asyncio.to_thread(sync_call)

async def evaluate_model(provider, model, prompt, semaphores, results, use_thinking, root, scale, duration):
    """ Evaluate a specific model with the given prompt and parameters.

    Args:
        provider (str): The name of the provider (e.g., "OpenAI", "Google").
        model (str): The model name.
        prompt (str): The prompt to use for generation.
        semaphores (dict): Dictionary of semaphores for rate limiting.
        results (list): List to store results of evaluations.
        use_thinking (bool): Whether to use extended thinking.
        root (str): The musical root note.
        scale (str): The musical scale.
        duration (str): The note duration.
    
    Returns:
        None: Appends the result to the results list and saves generation messages.
    """
    async with semaphores[provider]:
        # Asynchronously calling the model API
        try:
            midi_data, messages, cost, time_elapsed = await call_model_async(
                prompt=prompt,
                model_choice=model,
                temp=0.3,
                use_thinking=use_thinking,
                translate_prompt_choice=False
            )
        except Exception as e:
            midi_data = None
            messages = [str(e)]
            cost = 0
            time_elapsed = 0

        # Taking the generated loop and converting it to MIDI
        midi_file = MidiFile()
        if provider == "Google":
            loop_to_midi(midi_file, midi_data, times_as_string=True)
        else:
            loop_to_midi(midi_file, midi_data, times_as_string=False)

        os.makedirs(os.path.join("MIDI", model), exist_ok=True)
        safe_name = f"{prompt.replace(' ', '_')}_{use_thinking}.mid"
        midi_file.save(os.path.join("MIDI", model, safe_name))        
        
        # Run tests on the generated MIDI file
        test_results = tests.run_midi_tests(midi_file, root, scale, duration)
        result = {
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

        # Log and save the results
        results.append(result)
        append_log(result)
        save_generation_messages(provider, model, use_thinking, root, scale, duration, messages)

async def main():
    """ Main function to orchestrate the evaluation of models across multiple prompts and configurations. """

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
    total_prompt_count = len(prompts)

    models_by_provider = {
        "OpenAI": list(model_info["models"]["OpenAI"].keys()),
        "Google": list(model_info["models"]["Google"].keys()),
        "Anthropic": list(model_info["models"]["Anthropic"].keys())
    }

    # Compute expected total tests per (provider, model, use_thinking)
    expected_tests = {}  # key: (provider, model, use_thinking) -> expected_count
    for provider, models in models_by_provider.items():
        for model in models:
            ext_think = model_info["models"][provider][model]["extended_thinking"]
            # Always test without thinking
            expected_tests[(provider, model, False)] = total_prompt_count
            # If supported AND not OpenAI (they don't support toggling reasoning off), add thinking runs
            if ext_think and provider != "OpenAI":
                expected_tests[(provider, model, True)] = total_prompt_count

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

    # Live table that shows per-model aggregated progress
    table = Table(title="Model Evaluation Progress")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Thinking")
    table.add_column("Tested")
    table.add_column("Total")
    table.add_column("Pass Rate")
    table.add_column("Avg Latency (s)")
    table.add_column("Avg Cost")

    async def runner():
        """ Runner function to execute model evaluations concurrently. """
        tasks = []
        for prompt, root, scale, duration in prompts:
            for provider, models in models_by_provider.items():
                for model in models:
                    # Always test without thinking
                    tasks.append(evaluate_model(provider, model, prompt, semaphores, results, use_thinking=False, root=root, scale=scale, duration=duration))

                    # If extended_thinking is supported, also test with thinking
                    ext_think = model_info["models"][provider][model]["extended_thinking"]
                    if ext_think and provider != "OpenAI": 
                        tasks.append(evaluate_model(provider, model, prompt, semaphores, results, use_thinking=True, root=root, scale=scale, duration=duration))

        await asyncio.gather(*tasks)

    # Start the live table and run the evaluations
    with Live(table, console=console, refresh_per_second=4) as live:
        task = asyncio.create_task(runner())

        while not task.done():
            # Build a fresh table every refresh
            new_table = Table(title="Model Evaluation Progress")
            new_table.add_column("Provider")
            new_table.add_column("Model")
            new_table.add_column("Thinking")
            new_table.add_column("Tested")
            new_table.add_column("Total")
            new_table.add_column("Pass Rate")
            new_table.add_column("Avg Latency (s)")
            new_table.add_column("Avg Cost")

            # Aggregate current results into stats per (provider, model, use_thinking)
            stats = {}
            for r in results:
                key = (r["provider"], r["model"], r["use_thinking"])
                s = stats.setdefault(key, {"tested": 0, "passes": 0, "latency_sum": 0.0, "cost_sum": 0.0})
                s["tested"] += 1
                s["passes"] += 1 if r["output_pass"] else 0
                s["latency_sum"] += r.get("api_latency", 0.0) or 0.0
                s["cost_sum"] += r.get("cost", 0.0) or 0.0

            for key, total in expected_tests.items():
                provider, model, use_thinking = key
                s = stats.get(key, {"tested": 0, "passes": 0, "latency_sum": 0.0, "cost_sum": 0.0})
                tested = s["tested"]
                passes = s["passes"]
                avg_latency = (s["latency_sum"] / tested) if tested else 0.0
                avg_cost = (s["cost_sum"] / tested) if tested else 0.0
                pass_rate = (passes / tested * 100.0) if tested else 0.0

                new_table.add_row(
                    provider,
                    model,
                    "Yes" if use_thinking else "No",
                    f"{tested}",
                    f"{total}",
                    f"{pass_rate:.1f}%",
                    f"{avg_latency:.2f}",
                    f"{avg_cost:.4f}"
                )

            live.update(new_table)
            await asyncio.sleep(0.25)

        # ensure final results are shown once all tasks complete
        await task
        
        live.update(new_table)

if __name__ == "__main__":
    asyncio.run(main())
