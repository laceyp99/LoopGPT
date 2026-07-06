from types import SimpleNamespace

import pytest

from conductor_core.providers import google as gemini_api


def test_calc_cost_uses_reported_cached_tokens_without_storage_estimate():
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=200,
        cached_content_token_count=400,
    )

    cost = gemini_api.calc_cost("gemini-2.5-pro", usage)

    expected = (600 * 1.25 / 1_000_000) + (200 * 10.00 / 1_000_000) + (400 * 0.125 / 1_000_000)
    assert cost == pytest.approx(expected)


def test_calc_cost_clamps_cached_tokens_to_avoid_negative_input_cost():
    usage = SimpleNamespace(
        prompt_token_count=100,
        candidates_token_count=0,
        cached_content_token_count=150,
    )

    cost = gemini_api.calc_cost("gemini-2.5-pro", usage)

    assert cost == pytest.approx(100 * 0.125 / 1_000_000)


def test_calc_cost_handles_models_without_cache_pricing():
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=200,
        cached_content_token_count=None,
    )

    cost = gemini_api.calc_cost("gemini-2.0-flash-lite", usage)

    expected = (1000 * 0.075 / 1_000_000) + (200 * 0.30 / 1_000_000)
    assert cost == pytest.approx(expected)
