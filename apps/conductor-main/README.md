# Conductor Main

The Gradio user interface for `conductor-core`. The app owns UI layout,
callbacks, visualization, prompt editing, and UI progress adaptation. Loop
generation and artifact persistence are delegated to `LoopGenerationEngine`.

From the repository root, install editable packages and launch the client:

```powershell
py -3.12 -m pip install -e ".\packages\conductor-core[providers,playback]"
py -3.12 -m pip install -e ".\apps\conductor-main"
conductor-main
```

The root `app.py` remains a compatibility launcher during the transition.
