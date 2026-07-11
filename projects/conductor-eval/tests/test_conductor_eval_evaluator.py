import json
from types import SimpleNamespace

from conductor_core import GenerationRequest
from mido import MidiFile

from conductor_eval import EvalEngineAdapter, Evaluator


class RecordingEngine:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return self.result


def test_eval_engine_adapter_delegates_generation_to_core(tmp_path):
    midi_path = tmp_path / "core-loop.mid"
    MidiFile().save(midi_path)
    core_result = SimpleNamespace(
        midi_path=str(midi_path),
        messages=[{"role": "assistant", "content": "loop"}],
        cost=0.125,
    )
    engine = RecordingEngine(core_result)
    adapter = EvalEngineAdapter(tmp_path / "core-artifacts", engine=engine)

    midi, messages, cost = adapter.generate(
        description="warm quarter-note arpeggio",
        key="C",
        scale="major",
        model="gpt-test",
        provider="OpenAI",
        temperature=0.2,
        use_thinking=True,
        effort="medium",
    )

    assert isinstance(midi, MidiFile)
    assert midi_path.exists()
    assert messages == core_result.messages
    assert cost == core_result.cost
    assert engine.requests == [
        GenerationRequest(
            key="C",
            scale="major",
            description="warm quarter-note arpeggio",
            model="gpt-test",
            provider="OpenAI",
            temperature=0.2,
            use_thinking=True,
            effort="medium",
            render_audio=False,
        )
    ]


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
