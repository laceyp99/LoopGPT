# Legacy Evaluation Compatibility

The evaluation framework has moved to the installable
[`conductor-eval`](../projects/conductor-eval/README.md) project. The Python files
in this directory are thin compatibility wrappers for existing
`evaluation.*` imports and direct script paths; new code should use
`conductor_eval`.

## Installation

From the repository root:

```powershell
py -3.12 -m pip install -e ".\packages\conductor-core[providers]"
py -3.12 -m pip install -e ".\projects\conductor-eval[dashboard,dev]"
```

## Primary API

```python
from conductor_eval import Evaluator

evaluator = Evaluator(temperature=0.0)
```

Legacy imports remain available during migration:

```python
from evaluation.evaluator import Evaluator  # Compatibility only
from evaluation.tests import duration_test, scale_test  # Compatibility only
```

Launch the dashboard through the installed package:

```powershell
py -3.12 -m conductor_eval.analysis
py -3.12 -m conductor_eval.analysis projects/conductor-eval/evaluations/<run>
```

Evaluation output defaults to `projects/conductor-eval/evaluations/`. See the
[Conductor Eval README](../projects/conductor-eval/README.md) for configuration,
output structure, dashboard usage, and safety guidance.

Running either `py -3.12 -m conductor_eval.evaluator` or the legacy
`py -3.12 evaluation/evaluator.py` path is guarded because its example can make
many paid provider calls. Do not run broad evaluations without explicit
approval.
