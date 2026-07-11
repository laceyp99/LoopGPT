"""Compatibility wrapper for the extracted :mod:`conductor_eval` dashboard."""

from conductor_eval import analysis as _analysis

__all__ = [name for name in dir(_analysis) if not name.startswith("_")]


def __getattr__(name):
    """Delegate legacy analysis attributes to :mod:`conductor_eval.analysis`."""
    return getattr(_analysis, name)


def __dir__():
    """Expose the delegated dashboard API to interactive tooling."""
    return sorted(set(globals()) | set(__all__))


if __name__ == "__main__":
    _analysis.main()
