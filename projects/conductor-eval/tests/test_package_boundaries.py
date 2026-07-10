from pathlib import Path

import conductor_eval
from conductor_eval import EvalEngineAdapter, Evaluator


def test_public_package_exports_evaluator_api():
    assert conductor_eval.EvalEngineAdapter is EvalEngineAdapter
    assert conductor_eval.Evaluator is Evaluator


def test_evaluator_default_output_is_under_conductor_eval(monkeypatch):
    monkeypatch.setattr(Evaluator, "_setup_logging", lambda self: None)

    evaluator = Evaluator()

    assert evaluator.output_dir == Path("projects/conductor-eval/evaluations")


def test_package_source_has_no_legacy_import_or_path_mutation():
    package_dir = Path(__file__).resolve().parents[1] / "src" / "conductor_eval"
    forbidden_fragments = (
        "sys.path",
        "from src",
        "import src",
    )

    for source_path in package_dir.glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment!r} found in {source_path}"


def test_legacy_evaluator_wrapper_reexports_package_api():
    from evaluation.evaluator import (
        EvalEngineAdapter as LegacyEvalEngineAdapter,
    )
    from evaluation.evaluator import Evaluator as LegacyEvaluator

    assert LegacyEvalEngineAdapter is EvalEngineAdapter
    assert LegacyEvaluator is Evaluator
