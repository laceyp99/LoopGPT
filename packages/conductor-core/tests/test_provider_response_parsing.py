import json
from types import SimpleNamespace

import pytest

from conductor_core import models as objects
from conductor_core.providers import anthropic as claude_api
from conductor_core.providers import google as gemini_api
from conductor_core.providers import ollama as ollama_api
from conductor_core.providers import openai as openai_api


def _loop_payload():
    bar = {
        "num": 1,
        "notes": [
            {
                "pitch": "C",
                "octave": 4,
                "velocity": 100,
                "time": {"start_beat": 1, "duration": 1},
            }
        ],
    }
    return {"Bar_1": bar, "Bar_2": {**bar, "num": 2}, "Bar_3": {**bar, "num": 3}, "Bar_4": {**bar, "num": 4}}


def _loop_g_payload():
    bar = {
        "num": 1,
        "notes": [
            {
                "pitch": "C",
                "octave": 4,
                "velocity": 100,
                "time": {"start_beat": "one", "duration": "one"},
            }
        ],
    }
    return {"Bar_1": bar, "Bar_2": {**bar, "num": 2}, "Bar_3": {**bar, "num": 3}, "Bar_4": {**bar, "num": 4}}


def _fail_save_messages(*args, **kwargs):
    raise AssertionError("provider adapters should not write message logs")


def _anthropic_completion(payload):
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage)),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(partial_json=payload)),
        SimpleNamespace(type="message_stop"),
    ]


def test_openai_extract_reasoning_ignores_missing_summary():
    response = SimpleNamespace(output=[SimpleNamespace(type="reasoning")])

    assert openai_api.extract_reasoning(response) == ""


def test_openai_calc_price_uses_reported_cached_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            input_tokens_details=SimpleNamespace(cached_tokens=400),
        )
    )

    cost = openai_api.calc_price("gpt-4o-mini", response)

    expected = (600 * 0.15 / 1_000_000) + (400 * 0.075 / 1_000_000) + (200 * 0.60 / 1_000_000)
    assert cost == pytest.approx(expected)


def test_openai_calc_price_clamps_malformed_cached_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=0,
            input_tokens_details=SimpleNamespace(cached_tokens=150),
        )
    )

    cost = openai_api.calc_price("gpt-4o-mini", response)

    assert cost == pytest.approx(100 * 0.075 / 1_000_000)


def test_claude_calc_price_uses_reported_cache_creation_and_reads():
    output = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation": 300,
        "cache_read": 400,
    }

    cost = claude_api.calc_price("claude-sonnet-4-5", output)

    expected = (
        (1000 * 3.00 / 1_000_000)
        + (200 * 15.00 / 1_000_000)
        + (300 * 3.75 / 1_000_000)
        + (400 * 0.30 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


def test_gemini_process_output_rejects_empty_candidates():
    response = SimpleNamespace(candidates=[])

    with pytest.raises(ValueError, match="Google response did not include any candidates"):
        gemini_api.process_output(response)


def test_gemini_process_output_rejects_missing_parts():
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])

    with pytest.raises(ValueError, match="Google response did not include generated content parts"):
        gemini_api.process_output(response)


def test_claude_loop_gen_omits_cache_control_for_short_system_prompt(monkeypatch):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(claude_api, "initialize_anthropic_client", lambda: fake_client)
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: "short system prompt")
    monkeypatch.setattr(claude_api.utils, "save_messages_to_json", _fail_save_messages)

    midi_loop, messages, cost = claude_api.loop_gen(
        "write a loop",
        "claude-sonnet-4-5",
    )

    assert isinstance(midi_loop, objects.Loop)
    assert "cache_control" not in captured["system"][0]
    assert messages[0] == {"role": "system", "content": "short system prompt"}
    assert cost > 0


def test_claude_loop_gen_adds_cache_control_for_large_system_prompt(monkeypatch):
    captured = {}
    payload = json.dumps(_loop_payload())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _anthropic_completion(payload)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    long_prompt = "x" * claude_api.ANTHROPIC_CACHE_CONTROL_MIN_CHARS
    monkeypatch.setattr(claude_api, "initialize_anthropic_client", lambda: fake_client)
    monkeypatch.setattr(claude_api.utils, "get_loop_prompt", lambda: long_prompt)
    monkeypatch.setattr(claude_api.utils, "save_messages_to_json", _fail_save_messages)

    claude_api.loop_gen("write a loop", "claude-sonnet-4-5")

    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_ollama_loop_gen_accepts_missing_thinking(monkeypatch):
    payload = json.dumps(_loop_payload())
    completion = SimpleNamespace(message=SimpleNamespace(content=payload))
    fake_client = SimpleNamespace(chat=lambda **kwargs: completion)

    monkeypatch.setattr(ollama_api, "initialize_ollama_client", lambda: fake_client)
    monkeypatch.setattr(ollama_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(ollama_api.utils, "save_messages_to_json", _fail_save_messages)

    midi_loop, messages, cost = ollama_api.loop_gen("write a loop", "llama3")

    assert isinstance(midi_loop, objects.Loop)
    assert messages[-1]["content"] == str(midi_loop)
    assert cost == 0


def test_openai_loop_gen_does_not_write_message_log(monkeypatch):
    response = SimpleNamespace(
        output=[],
        output_parsed=objects.Loop.model_validate(_loop_payload()),
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **kwargs: response)
    )

    monkeypatch.setattr(openai_api, "initialize_openai_client", lambda: fake_client)
    monkeypatch.setattr(openai_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(openai_api.utils, "save_messages_to_json", _fail_save_messages)

    midi_loop, messages, cost = openai_api.loop_gen("write a loop", "gpt-4o-mini")

    assert isinstance(midi_loop, objects.Loop)
    assert messages[-1]["content"] == str(midi_loop)
    assert cost == 0


def test_gemini_loop_gen_does_not_write_message_log(monkeypatch):
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=json.dumps(_loop_g_payload()), thought=False)]
                )
            )
        ],
        parsed=objects.Loop_G.model_validate(_loop_g_payload()),
        usage_metadata=SimpleNamespace(
            cached_content_token_count=0,
            prompt_token_count=0,
            candidates_token_count=0,
        ),
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )

    monkeypatch.setattr(gemini_api, "initialize_gemini_client", lambda: fake_client)
    monkeypatch.setattr(gemini_api.utils, "get_loop_prompt", lambda: "system prompt")
    monkeypatch.setattr(gemini_api.utils, "save_messages_to_json", _fail_save_messages)

    midi_loop, messages, cost = gemini_api.loop_gen(
        "write a loop",
        "gemini-3.1-flash-lite",
    )

    assert isinstance(midi_loop, objects.Loop_G)
    assert messages[-1]["content"] == json.dumps(_loop_g_payload())
    assert cost == 0


def test_ollama_loop_gen_rejects_missing_content(monkeypatch):
    completion = SimpleNamespace(message=SimpleNamespace())
    fake_client = SimpleNamespace(chat=lambda **kwargs: completion)

    monkeypatch.setattr(ollama_api, "initialize_ollama_client", lambda: fake_client)
    monkeypatch.setattr(ollama_api.utils, "get_loop_prompt", lambda: "system prompt")

    with pytest.raises(ValueError, match="Ollama response did not include generated content"):
        ollama_api.loop_gen("write a loop", "llama3")
