import os
import subprocess
import sys
from pathlib import Path


def test_core_import_does_not_load_ui_dashboard_or_eval_modules():
    core_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(core_src)
    code = (
        "import conductor_core, sys; "
        "blocked = ['gradio', 'dash', 'plotly', 'evaluation']; "
        "print({name: name in sys.modules for name in blocked})"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == (
        "{'gradio': False, 'dash': False, 'plotly': False, 'evaluation': False}"
    )
