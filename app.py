"""Compatibility launcher for the packaged Conductor Gradio client."""

from conductor_main.app import *  # noqa: F401,F403
from conductor_main.app import main


if __name__ == "__main__":
    main()
