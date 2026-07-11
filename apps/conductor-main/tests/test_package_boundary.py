import os
import subprocess
import sys
from pathlib import Path


def test_gradio_client_does_not_import_legacy_application_modules():
    package_root = Path(__file__).parents[1] / "src" / "conductor_main"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package_root.glob("*.py"))

    assert "from src" not in source
    assert "import src" not in source
    assert "from evaluation" not in source
    assert "import evaluation" not in source


def test_gradio_client_imports_without_loading_legacy_modules():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import conductor_main.app, sys; "
                "print({'src': 'src' in sys.modules, "
                "'evaluation': 'evaluation' in sys.modules})"
            ),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "{'src': False, 'evaluation': False}"
