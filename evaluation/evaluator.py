"""Compatibility wrapper for the extracted :mod:`conductor_eval` evaluator."""

from conductor_eval.evaluator import (
    DIRECT_EVALUATION_CONFIRMATION,
    EvalEngineAdapter,
    Evaluator,
    confirm_direct_evaluation,
    main,
)

__all__ = [
    "DIRECT_EVALUATION_CONFIRMATION",
    "EvalEngineAdapter",
    "Evaluator",
    "confirm_direct_evaluation",
    "main",
]


if __name__ == "__main__":
    main()
