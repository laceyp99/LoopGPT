from pathlib import Path
import runpy

import pytest


def test_direct_evaluator_run_aborts_before_creating_outputs(
    monkeypatch, tmp_path
):
    evaluator_path = (
        Path(__file__).resolve().parents[1] / "evaluation" / "evaluator.py"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(evaluator_path), run_name="__main__")

    assert exc_info.value.code == 1
    assert not (tmp_path / "runs" / "run.log").exists()


def test_direct_evaluation_confirmation_requires_exact_phrase():
    from evaluation.evaluator import (
        DIRECT_EVALUATION_CONFIRMATION,
        confirm_direct_evaluation,
    )

    assert confirm_direct_evaluation(lambda _prompt: "y") is False
    assert (
        confirm_direct_evaluation(
            lambda _prompt: DIRECT_EVALUATION_CONFIRMATION
        )
        is True
    )


def test_direct_evaluation_confirmation_handles_closed_stdin():
    from evaluation.evaluator import confirm_direct_evaluation

    def raise_eof(_prompt):
        raise EOFError

    assert confirm_direct_evaluation(raise_eof) is False
