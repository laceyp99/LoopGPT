from types import SimpleNamespace

import pytest

from src import gemini_api


def _stub_model_info():
    return {
        "models": {
            "Google": {
                "tiered-model": {
                    "cost": {
                        "input": {"<=200k": 2.00, ">200k": 4.00},
                        "cache": {"<=200k": 0.20, ">200k": 0.40, "storage hour": 4.50},
                        "output": {"<=200k": 12.00, ">200k": 18.00},
                    }
                },
                "flat-model": {
                    "cost": {
                        "input": 0.30,
                        "cache": {"text": 0.03, "storage hour": 1.00},
                        "output": 2.50,
                    }
                },
            }
        }
    }


@pytest.fixture
def stub_model_info(monkeypatch):
    monkeypatch.setattr(gemini_api.utils, "get_model_info", _stub_model_info)


def test_calc_cost_scales_tiered_cache_storage_by_cached_tokens(stub_model_info):
    usage = SimpleNamespace(
        prompt_token_count=10000,
        candidates_token_count=1000,
        cached_content_token_count=5000,
    )

    cost = gemini_api.calc_cost("tiered-model", usage)

    expected = (
        (5000 * (2.00 / 1_000_000))
        + (1000 * (12.00 / 1_000_000))
        + (5000 * (0.20 / 1_000_000))
        + (5000 * (4.50 / 1_000_000) * gemini_api.IMPLICIT_CACHE_STORAGE_TTL_HOURS)
    )
    assert cost == pytest.approx(expected)


def test_calc_cost_scales_flat_cache_storage_by_cached_tokens(stub_model_info):
    usage = SimpleNamespace(
        prompt_token_count=20000,
        candidates_token_count=400,
        cached_content_token_count=12000,
    )

    cost = gemini_api.calc_cost("flat-model", usage)

    expected = (
        (8000 * (0.30 / 1_000_000))
        + (400 * (2.50 / 1_000_000))
        + (12000 * (0.03 / 1_000_000))
        + (12000 * (1.00 / 1_000_000) * gemini_api.IMPLICIT_CACHE_STORAGE_TTL_HOURS)
    )
    assert cost == pytest.approx(expected)


def test_calc_cost_caps_cached_tokens_at_prompt_tokens(stub_model_info):
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=10,
        cached_content_token_count=1500,
    )

    cost = gemini_api.calc_cost("tiered-model", usage)

    expected = (
        (10 * (12.00 / 1_000_000))
        + (1000 * (0.20 / 1_000_000))
        + (1000 * (4.50 / 1_000_000) * gemini_api.IMPLICIT_CACHE_STORAGE_TTL_HOURS)
    )
    assert cost == pytest.approx(expected)
    assert cost >= 0
