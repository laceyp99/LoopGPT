from pathlib import Path
import os
import subprocess
import sys


def test_gradio_client_does_not_import_legacy_application_modules():
    package_root = Path(__file__).parents[1] / "src" / "conductor_main"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package_root.glob("*.py"))

    assert "from src" not in source
    assert "import src" not in source
    assert "from evaluation" not in source
    assert "import evaluation" not in source


def test_gradio_client_imports_without_loading_legacy_modules():
    repository_root = Path(__file__).parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(repository_root / "apps" / "conductor-main" / "src"),
            str(repository_root / "packages" / "conductor-core" / "src"),
        ]
    )
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
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "{'src': False, 'evaluation': False}"
