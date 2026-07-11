import pytest

from conductor_eval import (
    DIRECT_EVALUATION_CONFIRMATION,
    confirm_direct_evaluation,
)
from conductor_eval import evaluator as evaluator_module


def test_direct_evaluator_run_aborts_before_creating_outputs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    with pytest.raises(SystemExit) as exc_info:
        evaluator_module.main()

    assert exc_info.value.code == 1
    assert not (tmp_path / "evaluations").exists()


def test_direct_evaluation_confirmation_requires_exact_phrase():
    assert confirm_direct_evaluation(lambda _prompt: "y") is False
    assert confirm_direct_evaluation(lambda _prompt: DIRECT_EVALUATION_CONFIRMATION) is True


def test_direct_evaluation_confirmation_handles_closed_stdin():
    def raise_eof(_prompt):
        raise EOFError

    assert confirm_direct_evaluation(raise_eof) is False
