import pytest

from src import runs


def test_generate_midi_routes_to_ollama_and_forwards_temperature(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}},
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda force_refresh=True: {"available": True, "models": ["llama3"]},
    )

    def fake_loop_gen(prompt, model, temp=0.0):
        captured.update({"prompt": prompt, "model": model, "temp": temp})
        return "loop", ["message"], 0

    monkeypatch.setattr(runs.ollama_api, "loop_gen", fake_loop_gen)

    result = runs.generate_midi("llama3", "write a loop", temp=0.7)

    assert result == ("loop", ["message"], 0)
    assert captured == {"prompt": "write a loop", "model": "llama3", "temp": 0.7}


def test_generate_midi_routes_to_openai_and_forwards_effort(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {
            "models": {
                "OpenAI": {"gpt-4o-mini": {}},
                "Google": {},
                "Anthropic": {},
            }
        },
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda force_refresh=True: {"available": True, "models": []},
    )

    def fake_loop_gen(prompt, model, temp=0.0, effort=None):
        captured.update(
            {"prompt": prompt, "model": model, "temp": temp, "effort": effort}
        )
        return "loop", ["message"], 1.25

    monkeypatch.setattr(runs.openai_api, "loop_gen", fake_loop_gen)

    result = runs.generate_midi("gpt-4o-mini", "write a loop", temp=0.2, effort="high")

    assert result == ("loop", ["message"], 1.25)
    assert captured == {
        "prompt": "write a loop",
        "model": "gpt-4o-mini",
        "temp": 0.2,
        "effort": "high",
    }


def test_generate_midi_routes_to_gemini_and_forwards_reasoning_options(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {
            "models": {
                "OpenAI": {},
                "Google": {"gemini-2.5-pro": {}},
                "Anthropic": {},
            }
        },
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda force_refresh=True: {"available": True, "models": []},
    )

    def fake_loop_gen(prompt, model, temp=0.0, use_thinking=None, effort=None):
        captured.update(
            {
                "prompt": prompt,
                "model": model,
                "temp": temp,
                "use_thinking": use_thinking,
                "effort": effort,
            }
        )
        return "loop", ["message"], 2.5

    monkeypatch.setattr(runs.gemini_api, "loop_gen", fake_loop_gen)

    result = runs.generate_midi(
        "gemini-2.5-pro",
        "write a loop",
        temp=0.4,
        use_thinking=True,
        effort="medium",
    )

    assert result == ("loop", ["message"], 2.5)
    assert captured == {
        "prompt": "write a loop",
        "model": "gemini-2.5-pro",
        "temp": 0.4,
        "use_thinking": True,
        "effort": "medium",
    }


def test_generate_midi_routes_to_claude_and_forwards_reasoning_options(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {
            "models": {
                "OpenAI": {},
                "Google": {},
                "Anthropic": {"claude-sonnet-4-5": {}},
            }
        },
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda force_refresh=True: {"available": True, "models": []},
    )

    def fake_loop_gen(prompt, model, temp=0.0, use_thinking=False, effort="low"):
        captured.update(
            {
                "prompt": prompt,
                "model": model,
                "temp": temp,
                "use_thinking": use_thinking,
                "effort": effort,
            }
        )
        return "loop", ["message"], 3.75

    monkeypatch.setattr(runs.claude_api, "loop_gen", fake_loop_gen)

    result = runs.generate_midi(
        "claude-sonnet-4-5",
        "write a loop",
        temp=0.1,
        use_thinking=True,
        effort="high",
    )

    assert result == ("loop", ["message"], 3.75)
    assert captured == {
        "prompt": "write a loop",
        "model": "claude-sonnet-4-5",
        "temp": 0.1,
        "use_thinking": True,
        "effort": "high",
    }


def test_generate_midi_rejects_unknown_models_when_ollama_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}},
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda force_refresh=True: {"available": False, "models": []},
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid Model Selected\. If you intended to use Ollama, it is currently unavailable\.",
    ):
        runs.generate_midi("unknown-model", "write a loop")


def test_generate_midi_rejects_unknown_models_when_ollama_is_available(monkeypatch):
    monkeypatch.setattr(
        runs,
        "get_model_info",
        lambda: {"models": {"OpenAI": {}, "Google": {}, "Anthropic": {}}},
    )
    monkeypatch.setattr(
        runs.ollama_api,
        "get_ollama_status",
        lambda force_refresh=True: {"available": True, "models": []},
    )

    with pytest.raises(ValueError, match="Invalid Model Selected"):
        runs.generate_midi("unknown-model", "write a loop")