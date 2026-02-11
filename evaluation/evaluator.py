"""
Evaluator class for unified MIDI loop generation testing across models.

This module provides a flexible evaluation framework that can:
- Test multiple prompts across multiple models
- Auto-detect test parameters from prompt text
- Support async execution for cloud providers and sync for local (Ollama)
- Save structured results including MIDI files, chat history, and test results
"""

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler
from mido import MidiFile
from pathlib import Path
from datetime import datetime
from typing import Union
import logging
import asyncio
import json
import time
import sys
import os

# Imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.runs as runs
import src.ollama_api as ollama_api
from src.midi_processing import loop_to_midi
from evaluation.tests import scale_test, duration_test

class Evaluator:
    """
    Unified evaluation framework for testing MIDI loop generation across models.

    Usage:
        evaluator = Evaluator()
        results = evaluator.evaluate(
            prompts="an arpeggiator using only quarter notes",
            roots=["C", "G"],
            models="openai",
            run_name="quarter_arp_test"
        )

    Attributes:
        SCALES: List of scales to test (always major and minor)
        EFFORT_LEVELS: Available effort levels for reasoning models
        AVAILABLE_TESTS: Registry of available test functions
        DURATION_KEYWORDS: Mapping of keywords to duration values for auto-detection
    """

    SCALES = ["major", "minor"]

    AVAILABLE_TESTS = {
        "scale": scale_test,
        "duration": duration_test,
    }

    DURATION_KEYWORDS = {
        "sixteenth": "sixteenth",
        "16th": "sixteenth",
        "eighth": "eighth",
        "8th": "eighth",
        "quarter": "quarter",
        "half": "half",
        "whole": "whole",
    }

    def __init__(self, output_dir: str = "evaluations", temperature: float = 0.0):
        """
        Initialize the Evaluator.

        Args:
            output_dir: Base directory for all evaluation outputs.
            temperature: Default temperature for generation.
        """
        self.output_dir = Path(output_dir)
        self.temperature = temperature
        self.console = Console(force_terminal=True)
        self._setup_logging()

        # Load model info
        with open("model_list.json", "r") as f:
            self.model_info = json.load(f)

    def _setup_logging(self):
        log_path = self.output_dir / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # 🔑 Remove any existing console handlers
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)

    def evaluate(self, prompts: Union[str, list[str]], roots: list[str], models: Union[str, list[str]] = "all", run_name: str = None,
        tests: list[str] = ["scale", "duration"], test_reasoning: bool = False, test_prompt_translation: bool = False) -> dict:
        """
        Run evaluation across all specified combinations.

        Args:
            prompts: Complete prompt(s) - will have " in {root} {scale}" appended.
            roots: List of root notes to test (e.g., ["C", "D", "F#"]).
            models: "all" | provider name ("openai", "ollama", etc.) | list of model names.
            run_name: Name for this evaluation run (used in output directory). Required.
            tests: List of test names to run (default: ["scale", "duration"]).
                   "scale" always runs. Others auto-detect params from prompt.
            test_reasoning: If True, test all thinking modes and effort levels for compatible models.
            test_prompt_translation: If True, test both with and without prompt translation.

        Returns:
            dict: Summary of evaluation results.

        Raises:
            ValueError: If run_name is not provided.
        """
        if run_name is None:
            raise ValueError("run_name is required")

        # Normalize prompts to list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Resolve models to (provider, model) tuples
        resolved_models = self._resolve_models(models)

        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = self.output_dir / f"{timestamp}_{run_name}"
        run_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "run_name": run_name,
            "timestamp": timestamp,
            "prompts": prompts,
            "roots": roots,
            "scales": self.SCALES,
            "models": [(p, m) for p, m in resolved_models],
            "tests": tests,
            "test_reasoning": test_reasoning,
            "test_prompt_translation": test_prompt_translation,
            "temperature": self.temperature,
        }
        with open(run_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Generate all task combinations
        tasks = self._generate_tasks(
            prompts=prompts,
            roots=roots,
            resolved_models=resolved_models,
            tests=tests,
            test_reasoning=test_reasoning,
            test_prompt_translation=test_prompt_translation,
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Starting evaluation '{run_name}' with {len(tasks)} total tasks")

        # Separate async and sync tasks
        async_tasks = [t for t in tasks if self._is_async_provider(t["provider"])]
        sync_tasks = [t for t in tasks if not self._is_async_provider(t["provider"])]

        all_results = []

        # Run async tasks (cloud providers)
        if async_tasks:
            logger.info(f"Running {len(async_tasks)} async tasks (cloud providers)")
            async_results = asyncio.run(
                self._run_async_batch(async_tasks, run_path, tests)
            )
            all_results.extend(async_results)

        # Run sync tasks (Ollama)
        if sync_tasks:
            logger.info(f"Running {len(sync_tasks)} sync tasks (Ollama)")
            sync_results = self._run_sync_batch(sync_tasks, run_path, tests)
            all_results.extend(sync_results)

        # Generate and save summary
        summary = self._generate_summary(all_results, config)
        with open(run_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {run_path}")
        return summary

    def run_tests(self, midi_data: MidiFile, root: str, scale: str, prompt: str, tests: list[str]) -> dict:
        """
        Run specified tests on MIDI data.

        Args:
            midi_data: The MIDI file object to test.
            root: Root note used in generation.
            scale: Scale used in generation.
            prompt: Original prompt (for parameter detection).
            tests: List of test names to run.

        Returns:
            dict: Test results with format:
                {
                    "scale": {...results...},
                    "duration": {...results...} or {"skipped": "reason"},
                    "overall_pass": bool
                }
        """
        results = {}
        all_passed = True

        for test_name in tests:
            if test_name not in self.AVAILABLE_TESTS:
                results[test_name] = {"skipped": f"Unknown test: {test_name}"}
                continue

            test_func = self.AVAILABLE_TESTS[test_name]

            if test_name == "scale":
                # Scale test always runs with provided root/scale
                try:
                    test_result = test_func(midi_data, root, scale)
                    test_result["ran"] = True
                    test_result["params"] = {"root": root, "scale": scale}
                    results[test_name] = test_result
                    if test_result.get("incorrect", 0) > 0:
                        all_passed = False
                except Exception as e:
                    results[test_name] = {"ran": False, "error": str(e)}
                    all_passed = False

            elif test_name == "duration":
                # Duration test requires auto-detection from prompt
                detected_params = self._detect_test_params(prompt, test_name)
                if "duration" not in detected_params:
                    results[test_name] = {
                        "ran": False,
                        "skipped": "No duration keyword detected in prompt",
                    }
                else:
                    try:
                        duration_value = detected_params["duration"]
                        test_result = test_func(midi_data, duration_value)
                        test_result["ran"] = True
                        test_result["params"] = {"duration": duration_value}
                        test_result["detected_from_prompt"] = True
                        results[test_name] = test_result
                        if test_result.get("incorrect", 0) > 0:
                            all_passed = False
                    except Exception as e:
                        results[test_name] = {"ran": False, "error": str(e)}
                        all_passed = False

            else:
                # Future tests can be added here with their own logic
                detected_params = self._detect_test_params(prompt, test_name)
                try:
                    test_result = test_func(midi_data, **detected_params)
                    test_result["ran"] = True
                    test_result["params"] = detected_params
                    results[test_name] = test_result
                    if test_result.get("incorrect", 0) > 0:
                        all_passed = False
                except Exception as e:
                    results[test_name] = {"ran": False, "error": str(e)}
                    all_passed = False

        results["overall_pass"] = all_passed
        return results

    def _resolve_models(self, models: Union[str, list[str]]) -> list[tuple[str, str]]:
        """
        Resolve model specification to (provider, model_name) tuples.

        Args:
            models: "all" | provider name | list of model names

        Returns:
            list: List of (provider, model_name) tuples
        """
        resolved = []

        if isinstance(models, str):
            models_lower = models.lower()
            logger = logging.getLogger(__name__)
            if models_lower == "all":
                # All cloud models from model_list.json
                for provider in ["OpenAI", "Anthropic", "Google"]:
                    if provider in self.model_info["models"]:
                        for model in self.model_info["models"][provider].keys():
                            resolved.append((provider, model))
                # All Ollama models
                try:
                    for model in ollama_api.model_list:
                        resolved.append(("Ollama", model))
                except Exception:
                    logger.warning("Could not load Ollama models")

            elif models_lower == "openai":
                for model in self.model_info["models"]["OpenAI"].keys():
                    resolved.append(("OpenAI", model))

            elif models_lower == "anthropic":
                for model in self.model_info["models"]["Anthropic"].keys():
                    resolved.append(("Anthropic", model))

            elif models_lower == "google":
                for model in self.model_info["models"]["Google"].keys():
                    resolved.append(("Google", model))

            elif models_lower == "ollama":
                try:
                    for model in ollama_api.model_list:
                        resolved.append(("Ollama", model))
                except Exception:
                    logger.warning("Could not load Ollama models")

            else:
                # Assume it's a single model name
                provider = self._get_provider(models)
                if provider:
                    resolved.append((provider, models))
                else:
                    raise ValueError(f"Unknown model or provider: {models}")

        elif isinstance(models, list):
            for model in models:
                provider = self._get_provider(model)
                if provider:
                    resolved.append((provider, model))
                else:
                    logger.warning(f"Unknown model: {model}, skipping")

        return resolved

    def _get_provider(self, model: str) -> str:
        """
        Determine provider for a given model name.

        Args:
            model: Model name string

        Returns:
            str: Provider name or None if not found
        """
        # Check cloud providers first
        for provider in ["OpenAI", "Anthropic", "Google"]:
            if provider in self.model_info["models"]:
                if model in self.model_info["models"][provider]:
                    return provider

        # Check Ollama
        try:
            if model in ollama_api.model_list:
                return "Ollama"
        except Exception:
            pass

        return None

    def _is_async_provider(self, provider: str) -> bool:
        """
        Return True if provider should use async execution.

        Args:
            provider: Provider name

        Returns:
            bool: True for cloud providers, False for Ollama
        """
        return provider in ["OpenAI", "Anthropic", "Google"]

    def _get_model_capabilities(self, provider: str, model: str) -> dict:
        """
        Get thinking/effort capabilities for a model.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            dict: Capabilities including extended_thinking, etc.
        """
        if provider == "Ollama":
            return {"extended_thinking": False, "effort_options": []}

        if provider in self.model_info["models"]:
            if model in self.model_info["models"][provider]:
                return self.model_info["models"][provider][model]

        return {"extended_thinking": False, "effort_options": []}

    def _detect_test_params(self, prompt: str, test_name: str) -> dict:
        """
        Extract test parameters from prompt text using keyword matching.

        Args:
            prompt: The full prompt string
            test_name: Name of the test

        Returns:
            dict: Detected parameters, or empty dict if not found
        """
        if test_name == "duration":
            prompt_lower = prompt.lower()
            for keyword, duration_value in self.DURATION_KEYWORDS.items():
                if keyword in prompt_lower:
                    return {"duration": duration_value}
            return {}

        return {}

    def _generate_tasks(self, prompts: list[str], roots: list[str], resolved_models: list[tuple[str, str]], tests: list[str],
        test_reasoning: bool, test_prompt_translation: bool) -> list[dict]:
        """
        Generate all task combinations to run.

        Args:
            prompts: List of base prompts
            roots: List of root notes
            resolved_models: List of (provider, model) tuples
            tests: List of test names
            test_reasoning: Whether to test reasoning variations
            test_prompt_translation: Whether to test translation variations

        Returns:
            list: List of task dictionaries
        """
        tasks = []

        for prompt in prompts:
            for root in roots:
                for scale in self.SCALES:
                    full_prompt = f"{prompt} in {root} {scale}"

                    for provider, model in resolved_models:
                        variations = self._generate_variations(
                            model=model,
                            provider=provider,
                            test_reasoning=test_reasoning,
                            test_prompt_translation=test_prompt_translation,
                        )

                        for variation in variations:
                            tasks.append(
                                {
                                    "provider": provider,
                                    "model": model,
                                    "original_prompt": prompt,
                                    "full_prompt": full_prompt,
                                    "root": root,
                                    "scale": scale,
                                    "use_thinking": variation["use_thinking"],
                                    "effort": variation["effort"],
                                    "translate_prompt": variation["translate_prompt"],
                                    "variation_name": variation["name"],
                                }
                            )

        return tasks

    def _generate_variations(self, model: str, provider: str, test_reasoning: bool, test_prompt_translation: bool) -> list[dict]:
        """
        Generate all config variations to test for a model.

        Args:
            model: Model name
            provider: Provider name
            test_reasoning: Whether to test reasoning variations
            test_prompt_translation: Whether to test translation variations

        Returns:
            list: List of variation config dictionaries
        """
        variations = []
        capabilities = self._get_model_capabilities(provider, model)
        supports_thinking = capabilities.get("extended_thinking", False)
        effort_options = capabilities.get("effort_options", [])

        # Base translation options
        translation_options = [False]
        if test_prompt_translation:
            translation_options = [False, True]

        for translate in translation_options:
            translate_suffix = "_translated" if translate else ""

            if test_reasoning and supports_thinking:
                # For OpenAI reasoning models (o-series), only effort levels matter
                if provider == "OpenAI" and supports_thinking:
                    for effort in effort_options:
                        variations.append(
                            {
                                "use_thinking": True,  # Always on for o-series
                                "effort": effort,
                                "translate_prompt": translate,
                                "name": f"{effort}{translate_suffix}",
                            }
                        )
                # For Anthropic/Google, test thinking with effort levels
                elif provider in ["Anthropic", "Google"] and effort_options:
                    # With thinking + effort levels
                    for effort in effort_options:
                        variations.append(
                            {
                                "use_thinking": True,
                                "effort": effort,
                                "translate_prompt": translate,
                                "name": f"{effort}{translate_suffix}",
                            }
                        )
                # For Anthropic/Google with thinking but no effort options
                elif provider in ["Anthropic", "Google"]:
                    # Without thinking
                    variations.append(
                        {
                            "use_thinking": False,
                            "effort": None,
                            "translate_prompt": translate,
                            "name": f"standard{translate_suffix}",
                        }
                    )
                    # With thinking, no effort specification
                    variations.append(
                        {
                            "use_thinking": True,
                            "effort": None,
                            "translate_prompt": translate,
                            "name": f"w_reasoning{translate_suffix}",
                        }
                    )
                else:
                    # Ollama or unknown - just basic variation
                    variations.append(
                        {
                            "use_thinking": False,
                            "effort": None,
                            "translate_prompt": translate,
                            "name": f"standard{translate_suffix}"
                        }
                    )
            else:
                # No reasoning testing - just basic run
                # For models with extended_thinking, default to least effort
                if supports_thinking and provider == "OpenAI":
                    variations.append(
                        {
                            "use_thinking": True,
                            "effort": effort_options[0],
                            "translate_prompt": translate,
                            "name": f"{effort_options[0]}{translate_suffix}"
                        }
                    )
                else:
                    variations.append(
                        {
                            "use_thinking": False,
                            "effort": None,
                            "translate_prompt": translate,
                            "name": f"standard{translate_suffix}"
                        }
                    )

        return variations

    async def _run_async_batch(self, tasks: list[dict], run_path: Path, tests_to_run: list[str]) -> list[dict]:
        """
        Run tasks asynchronously with rate limiting.

        Args:
            tasks: List of task dictionaries
            run_path: Path to save results
            tests_to_run: List of test names to run

        Returns:
            list: List of result dictionaries
        """
        # Build semaphores from RPM
        semaphores = {}
        for provider in ["OpenAI", "Anthropic", "Google"]:
            if provider in self.model_info["models"]:
                rpms = []
                for model in self.model_info["models"][provider].keys():
                    rate_info = self.model_info["models"][provider][model].get("rate_limits", {})
                    rpm = rate_info.get("RPM", 60)
                    rpms.append(rpm)
                max_concurrent = max(1, min(rpms) // 60) if rpms else 1
                semaphores[provider] = asyncio.Semaphore(max_concurrent)

        results = []
        total_tasks = len(tasks)

        # Create live table for progress
        table = Table(title="Evaluation Progress")
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("Variation")
        table.add_column("Progress")
        table.add_column("Pass Rate")
        table.add_column("Avg Latency")
        table.add_column("Avg Cost")

        async def run_single_task(task: dict) -> dict:
            provider = task["provider"]
            async with semaphores.get(provider, asyncio.Semaphore(1)):
                return await asyncio.to_thread(
                    self._run_single,
                    task=task,
                    run_path=run_path,
                    tests_to_run=tests_to_run,
                )

        with Live(table, console=self.console, refresh_per_second=2) as live:
            # Create all async tasks
            async_tasks = [run_single_task(t) for t in tasks]

            # Run with gathering and update progress
            for coro in asyncio.as_completed(async_tasks):
                result = await coro
                results.append(result)

                # Update table
                new_table = Table(
                    title=f"Evaluation Progress ({len(results)}/{total_tasks})"
                )
                new_table.add_column("Provider")
                new_table.add_column("Model")
                new_table.add_column("Tested")
                new_table.add_column("Pass Rate")
                new_table.add_column("Avg Latency")
                new_table.add_column("Avg Cost")

                # Aggregate stats by model
                stats = {}
                for r in results:
                    key = (r["provider"], r["model"])
                    s = stats.setdefault(
                        key,
                        {
                            "tested": 0,
                            "passed": 0,
                            "latency_sum": 0.0,
                            "cost_sum": 0.0,
                        },
                    )
                    s["tested"] += 1
                    if r.get("tests", {}).get("overall_pass", False):
                        s["passed"] += 1
                    s["latency_sum"] += r.get("metrics", {}).get("api_latency", 0.0)
                    s["cost_sum"] += r.get("metrics", {}).get("cost", 0.0)

                for (provider, model), s in stats.items():
                    pass_rate = (s["passed"] / s["tested"] * 100) if s["tested"] else 0
                    avg_latency = s["latency_sum"] / s["tested"] if s["tested"] else 0
                    avg_cost = s["cost_sum"] / s["tested"] if s["tested"] else 0
                    new_table.add_row(
                        provider,
                        model,
                        str(s["tested"]),
                        f"{pass_rate:.1f}%",
                        f"{avg_latency:.2f}s",
                        f"${avg_cost:.4f}",
                    )

                live.update(new_table)

        return results

    def _run_sync_batch(self, tasks: list[dict], run_path: Path, tests_to_run: list[str]) -> list[dict]:
        """
        Run tasks synchronously (for Ollama).

        Args:
            tasks: List of task dictionaries
            run_path: Path to save results
            tests_to_run: List of test names to run

        Returns:
            list: List of result dictionaries
        """
        # Sort tasks by model to minimize GPU memory swaps
        tasks = sorted(tasks, key=lambda t: t["model"])

        results = []
        total_tasks = len(tasks)

        table = Table(title="Ollama Evaluation Progress")
        table.add_column("Model")
        table.add_column("Progress")
        table.add_column("Pass Rate")
        table.add_column("Avg Latency")

        with Live(table, console=self.console, refresh_per_second=2) as live:
            for i, task in enumerate(tasks):
                result = self._run_single(task, run_path, tests_to_run)
                results.append(result)

                # Update table
                new_table = Table(
                    title=f"Evaluation Progress ({len(results)}/{total_tasks})"
                )
                new_table.add_column("Model")
                new_table.add_column("Tested")
                new_table.add_column("Pass Rate")
                new_table.add_column("Avg Latency")

                # Aggregate stats by model
                stats = {}
                for r in results:
                    key = r["model"]
                    s = stats.setdefault(
                        key,
                        {
                            "tested": 0,
                            "passed": 0,
                            "latency_sum": 0.0,
                        },
                    )
                    s["tested"] += 1
                    if r.get("tests", {}).get("overall_pass", False):
                        s["passed"] += 1
                    s["latency_sum"] += r.get("metrics", {}).get("api_latency", 0.0)

                for model, s in stats.items():
                    pass_rate = (s["passed"] / s["tested"] * 100) if s["tested"] else 0
                    avg_latency = s["latency_sum"] / s["tested"] if s["tested"] else 0
                    new_table.add_row(
                        model,
                        str(s["tested"]),
                        f"{pass_rate:.1f}%",
                        f"{avg_latency:.2f}s",
                    )

                live.update(new_table)

        return results

    def _run_single(self, task: dict, run_path: Path, tests_to_run: list[str]) -> dict:
        """
        Run single generation, tests, and save results.

        Args:
            task: Task dictionary with all parameters
            run_path: Path to save results
            tests_to_run: List of test names to run

        Returns:
            dict: Result dictionary
        """
        provider = task["provider"]
        model = task["model"]
        full_prompt = task["full_prompt"]
        original_prompt = task["original_prompt"]
        root = task["root"]
        scale = task["scale"]
        use_thinking = task["use_thinking"]
        effort = task["effort"]
        translate_prompt = task["translate_prompt"]
        variation_name = task["variation_name"]

        # Build result structure
        result = {
            "model": model,
            "provider": provider,
            "prompt": full_prompt,
            "original_prompt": original_prompt,
            "root": root,
            "scale": scale,
            "config": {
                "use_thinking": use_thinking,
                "effort": effort,
                "translate_prompt": translate_prompt,
                "temperature": self.temperature,
            },
            "metrics": {
                "api_latency": 0.0,
                "cost": 0.0,
            },
            "tests": {},
            "error": None,
        }
        logger = logging.getLogger(__name__)
        # Generate MIDI
        try:
            start_time = time.perf_counter()
            loop, messages, cost = runs.generate_midi(
                model_choice=model,
                prompt=full_prompt,
                temp=self.temperature,
                translate_prompt_choice=translate_prompt,
                use_thinking=use_thinking,
                effort=effort if effort else "low",
            )
            time_elapsed = time.perf_counter() - start_time

            result["metrics"]["api_latency"] = time_elapsed
            result["metrics"]["cost"] = cost

        except Exception as e:
            logger.error(f"Generation failed for {model}: {e}")
            result["error"] = str(e)
            result["tests"]["overall_pass"] = False
            # Still save the result even on failure
            self._save_results(result, None, [], run_path, task)
            return result

        # Convert loop to MIDI
        midi_file = MidiFile()
        try:
            if provider == "Google":
                loop_to_midi(midi_file, loop, times_as_string=True)
            else:
                loop_to_midi(midi_file, loop, times_as_string=False)
        except Exception as e:
            logger.error(f"MIDI conversion failed for {model}: {e}")
            result["error"] = f"MIDI conversion error: {e}"
            result["tests"]["overall_pass"] = False
            self._save_results(result, None, messages, run_path, task)
            return result

        # Run tests
        test_results = self.run_tests(
            midi_data=midi_file,
            root=root,
            scale=scale,
            prompt=original_prompt,
            tests=tests_to_run,
        )
        result["tests"] = test_results

        # Save results
        self._save_results(result, midi_file, messages, run_path, task)

        return result

    def _save_results(self, result: dict, midi_data: MidiFile, messages: list, run_path: Path, task: dict) -> None:
        """
        Save MIDI, messages, and test results to disk.

        Args:
            result: Result dictionary
            midi_data: MIDI file object (or None on failure)
            messages: Chat messages list
            run_path: Base path for this run
            task: Task dictionary with path info
        """
        provider = task["provider"]
        model = task["model"]
        original_prompt = task["original_prompt"]
        root = task["root"]
        scale = task["scale"]
        variation_name = task["variation_name"]

        # Create directory structure
        prompt_slug = self._sanitize_filename(original_prompt, max_len=50)
        result_dir = (
            run_path / "results" / provider / model.replace(":", "") / prompt_slug / f"{root}_{scale}"
        )

        # Add variation folder if there are multiple variations
        if variation_name != "standard":
            result_dir = result_dir / variation_name

        result_dir.mkdir(parents=True, exist_ok=True)

        # Save MIDI
        if midi_data is not None:
            midi_path = result_dir / "output.mid"
            midi_data.save(str(midi_path))

        # Save messages (for fine-tuning)
        messages_path = result_dir / "messages.json"
        with open(messages_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)

        # Save test results
        results_path = result_dir / "test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def _generate_summary(self, all_results: list[dict], config: dict) -> dict:
        """
        Aggregate results into summary statistics.

        Args:
            all_results: List of all result dictionaries
            config: Original configuration dictionary

        Returns:
            dict: Summary with aggregated statistics
        """
        summary = {
            "run_id": f"{config['timestamp']}_{config['run_name']}",
            "config": config,
            "totals": {
                "total_generations": len(all_results),
                "successful_generations": 0,
                "failed_generations": 0,
                "overall_pass_count": 0,
                "overall_pass_rate": 0.0,
                "total_cost": 0.0,
                "total_time": 0.0,
            },
            "by_model": {},
            "by_root": {},
            "by_scale": {},
        }

        for r in all_results:
            # Totals
            if r.get("error"):
                summary["totals"]["failed_generations"] += 1
            else:
                summary["totals"]["successful_generations"] += 1

            if r.get("tests", {}).get("overall_pass", False):
                summary["totals"]["overall_pass_count"] += 1

            summary["totals"]["total_cost"] += r.get("metrics", {}).get("cost", 0.0)
            summary["totals"]["total_time"] += r.get("metrics", {}).get(
                "api_latency", 0.0
            )

            # By model
            model = r["model"]
            if model not in summary["by_model"]:
                summary["by_model"][model] = {
                    "provider": r["provider"],
                    "tested": 0,
                    "passed": 0,
                    "failed": 0,
                    "pass_rate": 0.0,
                    "total_cost": 0.0,
                    "total_latency": 0.0,
                    "avg_latency": 0.0,
                }
            m = summary["by_model"][model]
            m["tested"] += 1
            if r.get("tests", {}).get("overall_pass", False):
                m["passed"] += 1
            if r.get("error"):
                m["failed"] += 1
            m["total_cost"] += r.get("metrics", {}).get("cost", 0.0)
            m["total_latency"] += r.get("metrics", {}).get("api_latency", 0.0)

            # By root
            root = r["root"]
            if root not in summary["by_root"]:
                summary["by_root"][root] = {"tested": 0, "passed": 0}
            summary["by_root"][root]["tested"] += 1
            if r.get("tests", {}).get("overall_pass", False):
                summary["by_root"][root]["passed"] += 1

            # By scale
            scale = r["scale"]
            if scale not in summary["by_scale"]:
                summary["by_scale"][scale] = {"tested": 0, "passed": 0}
            summary["by_scale"][scale]["tested"] += 1
            if r.get("tests", {}).get("overall_pass", False):
                summary["by_scale"][scale]["passed"] += 1

        # Calculate rates
        total = summary["totals"]["total_generations"]
        if total > 0:
            summary["totals"]["overall_pass_rate"] = (
                summary["totals"]["overall_pass_count"] / total
            )

        for model, m in summary["by_model"].items():
            if m["tested"] > 0:
                m["pass_rate"] = m["passed"] / m["tested"]
                m["avg_latency"] = m["total_latency"] / m["tested"]

        for root, r in summary["by_root"].items():
            if r["tested"] > 0:
                r["pass_rate"] = r["passed"] / r["tested"]

        for scale, s in summary["by_scale"].items():
            if s["tested"] > 0:
                s["pass_rate"] = s["passed"] / s["tested"]

        return summary

    def _sanitize_filename(self, text: str, max_len: int = 50) -> str:
        """
        Create safe filename from text.

        Args:
            text: Original text
            max_len: Maximum length of output

        Returns:
            str: Sanitized filename
        """
        # Replace spaces and special characters
        safe = text.replace(" ", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "_-")
        return safe[:max_len]


if __name__ == "__main__":

    # with open('model_list.json', 'r') as f:
    #     model_info = json.load(f)
    
    # model_names = [m for provider in model_info['models'].values() for m in provider.keys()]
    # print(len(model_names))

    eval = Evaluator(output_dir="runs",temperature=0.0)
    eval.evaluate(
        prompts=[
            "An arpeggiator in only quarter notes",
            "An arpeggiator in only eighth notes",
            "An arpeggiator in only sixteenth notes"
        ],
        roots=["C", "A", "F#", "Eb"], 
        models="ollama",
        run_name="arpeggiator_local_pt", 
        tests=["scale", "duration"], 
        test_reasoning=False, 
        test_prompt_translation=True
    )