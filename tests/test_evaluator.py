import json

from mido import MidiFile

from evaluation.evaluator import Evaluator


def test_save_results_uses_per_result_loop_filename(tmp_path):
    evaluator = Evaluator(output_dir=str(tmp_path / "evaluations"))
    run_path = tmp_path / "run"
    midi = MidiFile()
    messages = [{"role": "user", "content": "prompt"}]
    result = {
        "model": "gpt-test",
        "provider": "OpenAI",
        "tests": {"overall_pass": True},
    }
    task = {
        "provider": "OpenAI",
        "model": "gpt-test",
        "original_prompt": "warm loop",
        "root": "C",
        "scale": "major",
        "variation_name": "standard",
    }

    evaluator._save_results(result, midi, messages, run_path, task)

    result_dir = run_path / "results" / "OpenAI" / "gpt-test" / "warm_loop" / "C_major"
    assert (result_dir / "loop.mid").exists()
    legacy_filename = "output" + ".mid"
    assert not (result_dir / legacy_filename).exists()
    assert json.loads((result_dir / "messages.json").read_text(encoding="utf-8")) == messages
