"""
This file is using Gradio for the LoopGPT application. It makes the generation progress more user friendly by providing a GUI for the user to interact with.

Features:
- Text to MIDI generation with multiple AI providers
- Audio playback of generated MIDI using FluidSynth
- Session history with persistent storage (up to 20 generations)
- Toggleable history sidebar panel
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import gradio as gr
from conductor_core import (
    EngineConfig,
    GenerationRequest,
    LoopGenerationEngine,
    ProviderCredentials,
)
from conductor_core.music import get_loop_prompt, get_model_info
from conductor_core.playback import (
    add_soundfont_search_dir,
    get_default_soundfont,
    get_playback_status_message,
    is_playback_available,
    list_soundfonts,
    midi_to_mp3,
)
from conductor_core.providers import ollama as ollama_api
from conductor_core.storage import FilesystemArtifactStore
from mido import MidiFile

from conductor_main.visualization import visualize_midi_plotly

DEFAULT_PROVIDER = "Google"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_app_data_dir() -> Path:
    """Resolve the app data directory (project-local by default)."""
    return Path(os.environ.get("CONDUCTOR_MAIN_DATA_DIR", PROJECT_ROOT)).expanduser()


APP_DATA_DIR = _resolve_app_data_dir()
APP_SOUNDFONT_DIR = Path(
    os.environ.get(
        "CONDUCTOR_MAIN_SOUNDFONT_DIR",
        PROJECT_ROOT / "soundfonts",
    )
).expanduser()
PROMPT_OVERRIDE_PATH = APP_DATA_DIR / "Prompts" / "loop gen.txt"

add_soundfont_search_dir(APP_SOUNDFONT_DIR)
HISTORY_STORE = FilesystemArtifactStore(APP_DATA_DIR / "generations")


def load_history():
    return HISTORY_STORE.load_history()


def get_generation(gen_id):
    return HISTORY_STORE.get_generation(gen_id)


def delete_generation(gen_id):
    return HISTORY_STORE.delete_generation(gen_id)


def update_generation_audio(gen_id, audio_path, soundfont=None):
    return HISTORY_STORE.update_generation_audio(gen_id, audio_path, soundfont=soundfont)


def format_price_summary(price_value):
    """Format scalar or tiered pricing for dropdown labels."""
    if isinstance(price_value, (int, float)):
        return f"${price_value:.2f}"

    if isinstance(price_value, dict):
        numeric_values = [
            value for value in price_value.values() if isinstance(value, (int, float))
        ]
        if not numeric_values:
            return None

        min_price = min(numeric_values)
        max_price = max(numeric_values)
        if min_price == max_price:
            return f"${min_price:.2f}"
        return f"${min_price:.2f}-${max_price:.2f}"

    return None


def format_model_label(provider, model_name):
    """Build the model dropdown label with pricing when available."""
    if provider == "Ollama":
        return model_name

    model_info = get_model_info()
    provider_models = model_info["models"].get(provider, {})
    model_data = provider_models.get(model_name, {})
    cost = model_data.get("cost")

    if not cost or "input" not in cost or "output" not in cost:
        return model_name

    input_price = format_price_summary(cost["input"])
    output_price = format_price_summary(cost["output"])

    if not input_price or not output_price:
        return model_name

    return f"{model_name} ({input_price} in / {output_price} out per 1M tokens)"


def get_model_dropdown_choices(provider):
    """Get dropdown choices as (label, value) tuples for a provider."""
    models = get_models_for_provider(provider)
    return [(format_model_label(provider, model_name), model_name) for model_name in models]


def get_providers():
    """Get list of available providers including Ollama if models are available.

    Returns:
        list: List of provider names.
    """
    model_info = get_model_info()
    providers = list(model_info["models"].keys())
    if ollama_api.get_ollama_status()["available"]:
        providers.append("Ollama")
    return providers


def get_models_for_provider(provider):
    """Get list of models for a specific provider.

    Args:
        provider (str): The provider name.

    Returns:
        list: List of model names for the provider.
    """
    if provider == "Ollama":
        return ollama_api.get_model_list()
    model_info = get_model_info()
    if provider in model_info["models"]:
        return list(model_info["models"][provider].keys())
    return []


def get_selected_model(provider, model_choice):
    """Normalize the selected model for a provider."""
    models = get_models_for_provider(provider)
    if model_choice in models:
        return model_choice
    return models[0] if models else None


def get_model_settings(provider, model_choice, use_thinking=False):
    """Resolve the provider/model UI settings for dependent controls."""
    selected_model = get_selected_model(provider, model_choice)
    if not selected_model or provider == "Ollama":
        return {
            "selected_model": selected_model,
            "show_temperature": True,
            "temperature_value": 0.1,
            "show_thinking": False,
            "thinking_value": False,
            "effort_options": [],
            "effort_value": "low",
            "show_effort": False,
        }

    model_info = get_model_info()
    model_config = model_info["models"][provider][selected_model]
    effort_options = model_config.get("effort_options", [])
    supports_toggle_reasoning = (
        provider in {"Anthropic", "Google"}
        and model_config.get("extended_thinking", False)
        and not effort_options
    )
    show_temperature = True
    temperature_value = 0.1

    if provider == "OpenAI" and model_config.get("extended_thinking", False):
        show_temperature = False
        temperature_value = 1.0
    elif provider in {"Anthropic", "Google"} and (
        effort_options or (supports_toggle_reasoning and use_thinking)
    ):
        show_temperature = False
        temperature_value = 1.0

    thinking_value = bool(use_thinking) if supports_toggle_reasoning else False
    effort_value = effort_options[0] if effort_options else "low"

    return {
        "selected_model": selected_model,
        "show_temperature": show_temperature,
        "temperature_value": temperature_value,
        "show_thinking": supports_toggle_reasoning,
        "thinking_value": thinking_value,
        "effort_options": effort_options,
        "effort_value": effort_value,
        "show_effort": bool(effort_options),
    }


def sync_model_capabilities(provider, model_choice, use_thinking=False):
    """Synchronize model selection and dependent controls from one explicit code path."""
    choices = get_model_dropdown_choices(provider)
    settings = get_model_settings(provider, model_choice, use_thinking)

    return (
        gr.update(choices=choices, value=settings["selected_model"]),
        gr.update(
            visible=settings["show_temperature"],
            value=settings["temperature_value"],
        ),
        gr.update(
            visible=settings["show_thinking"],
            value=settings["thinking_value"],
        ),
        gr.update(
            choices=settings["effort_options"] or None,
            value=settings["effort_value"],
            visible=settings["show_effort"],
        ),
    )


def sync_controls_for_provider(provider):
    """Reset dependent controls when the provider changes."""
    return sync_model_capabilities(provider, None, False)


def sync_controls_for_model(provider, model_choice):
    """Reset dependent controls when the selected model changes."""
    return sync_model_capabilities(provider, model_choice, False)


def sync_controls_for_thinking(provider, model_choice, use_thinking):
    """Refresh dependent controls when the reasoning toggle changes."""
    return sync_model_capabilities(provider, model_choice, use_thinking)


def get_soundfont_choices():
    """Get the available SoundFont filenames for the UI."""
    return list_soundfonts()


def get_selected_soundfont(soundfont_choice=None):
    """Normalize the selected SoundFont for the UI."""
    soundfonts = get_soundfont_choices()
    if not soundfonts:
        return None

    if soundfont_choice:
        requested_name = os.path.basename(soundfont_choice)
        if requested_name in soundfonts:
            return requested_name

    default_soundfont = get_default_soundfont()
    if default_soundfont:
        default_soundfont_name = os.path.basename(default_soundfont)
        if default_soundfont_name in soundfonts:
            return default_soundfont_name

    return soundfonts[0]


def get_soundfont_dropdown_update(soundfont_choice=None):
    """Build a dropdown update for the current SoundFont selection."""
    return gr.update(
        choices=get_soundfont_choices(),
        value=get_selected_soundfont(soundfont_choice),
    )


def has_active_rerender_target(midi_path):
    """Return whether the UI currently has a MIDI file available to rerender."""
    return bool(midi_path and os.path.exists(midi_path))


def rerender_available(soundfont_choice=None, midi_path=None):
    """Return whether rerendering should be enabled for the current UI state."""
    selected_soundfont = get_selected_soundfont(soundfont_choice)
    playback_available, _ = is_playback_available(selected_soundfont)
    return (
        playback_available
        and selected_soundfont is not None
        and has_active_rerender_target(midi_path)
    )


def get_rerender_button_update(soundfont_choice=None, midi_path=None):
    """Build a button update for the current rerender availability."""
    return gr.update(interactive=rerender_available(soundfont_choice, midi_path))


def get_soundfont_status_message(soundfont_choice=None):
    """Build the status text for the current SoundFont and playback state."""
    selected_soundfont = get_selected_soundfont(soundfont_choice)
    playback_available, _ = is_playback_available(selected_soundfont)

    if playback_available and selected_soundfont:
        return f"Found {len(get_soundfont_choices())} SoundFonts. Selected {selected_soundfont}."

    return get_playback_status_message(selected_soundfont)


def refresh_soundfont_controls(soundfont_choice=None, midi_path=None):
    """Refresh SoundFont UI controls from the filesystem."""
    return (
        get_soundfont_dropdown_update(soundfont_choice),
        get_rerender_button_update(soundfont_choice, midi_path),
        get_soundfont_status_message(soundfont_choice),
    )


def rerender_current_audio(
    midi_path,
    soundfont_choice,
    saved_soundfont,
    generation_id,
    current_audio_path,
):
    """Re-render the current MIDI file with the selected SoundFont on demand."""
    if not midi_path:
        return (
            current_audio_path,
            "No MIDI file available to re-render.",
            saved_soundfont,
            current_audio_path,
        )

    if not os.path.exists(midi_path):
        return (
            current_audio_path,
            f"MIDI file not found: {midi_path}",
            saved_soundfont,
            current_audio_path,
        )

    selected_soundfont = get_selected_soundfont(soundfont_choice)
    if not selected_soundfont:
        return (
            current_audio_path,
            get_playback_status_message(soundfont_choice),
            saved_soundfont,
            current_audio_path,
        )

    if (
        saved_soundfont == selected_soundfont
        and current_audio_path
        and os.path.exists(current_audio_path)
    ):
        return (
            current_audio_path,
            f"Audio already rendered with {selected_soundfont}.",
            saved_soundfont,
            current_audio_path,
        )

    output_path = current_audio_path or f"{os.path.splitext(midi_path)[0]}.mp3"
    rendered_audio_path = midi_to_mp3(
        midi_path,
        output_path=output_path,
        soundfont_name=selected_soundfont,
    )
    if rendered_audio_path is None:
        return (
            current_audio_path,
            get_playback_status_message(selected_soundfont),
            saved_soundfont,
            current_audio_path,
        )

    persisted_audio_path = rendered_audio_path
    if generation_id:
        updated_generation = update_generation_audio(
            generation_id,
            rendered_audio_path,
            soundfont=selected_soundfont,
        )
        if updated_generation and updated_generation.audio_path:
            persisted_audio_path = updated_generation.audio_path

    return (
        persisted_audio_path,
        f"Rendered audio with {selected_soundfont}.",
        selected_soundfont,
        persisted_audio_path,
    )


def load_app_prompt_override():
    """Load the app-owned prompt override if the user has saved one."""
    if not PROMPT_OVERRIDE_PATH.exists():
        return None

    with PROMPT_OVERRIDE_PATH.open("r", encoding="utf-8") as prompt_file:
        return prompt_file.read()


def get_prompt_editor_text():
    """Return the prompt text shown in the Prompt Editor."""
    return load_app_prompt_override() or get_loop_prompt()


def save_prompts(loop_gen_text):
    """This function saves any changes to the loop generation prompt to the text file.

    Args:
        loop_gen_text (str): The loop generation prompt text.

    Returns:
        str: A message indicating the status of the save operation.
    """
    PROMPT_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROMPT_OVERRIDE_PATH.open("w", encoding="utf-8") as f:
        f.write(loop_gen_text)
    return (
        "Prompts saved successfully at " + datetime.now().strftime("%I:%M:%S %p on %B %d, %Y") + "."
    )


def run_loop(
    key,
    scale,
    description,
    temp,
    model_choice,
    use_thinking,
    effort,
    soundfont_choice,
    openai_key,
    gemini_key,
    claude_key,
):
    """Run the loop generation process based on user inputs and selected model.

    This is a generator function that yields progress updates while the API call runs
    in a background thread. Gradio can stop waiting for the generator, but that does
    not cancel the in-flight provider request.

    Args:
        key (str): The key for the loop that the user selects from the dropdown.
        scale (str): The scale for the loop that the user selects from the Major/minor dropdown.
        description (str): A description of the loop that the user input in the text box.
        temp (float): The sampling temperature for the model that the user selects from the slider.
        model_choice (str): The model that the user selects from the dropdown.
        use_thinking (bool): Whether to enable extended thinking for supported Claude and Gemini models.
        effort (str): The reasoning effort level for supported OpenAI models.
         soundfont_choice (str): The selected SoundFont filename for audio rendering.
        openai_key (str): The OpenAI API key that the user inputs in the text box.
        gemini_key (str): The Gemini API key that the user inputs in the text box.
        claude_key (str): The Claude API key that the user inputs in the text box.

    Yields:
         tuple: (file_path, audio_path, visualization, status_message, stop_button_update,
             generation_id, saved_soundfont, current_audio_path) - intermediate yields
             show progress and keep the stop-waiting control visible, final yield contains the generated MIDI,
             audio, and persisted audio metadata for rerendering.
    """
    try:
        selected_soundfont = get_selected_soundfont(soundfont_choice)
        credentials = ProviderCredentials(
            openai_api_key=openai_key.strip() if openai_key and openai_key.strip() else None,
            google_api_key=gemini_key.strip() if gemini_key and gemini_key.strip() else None,
            anthropic_api_key=claude_key.strip() if claude_key and claude_key.strip() else None,
        )
        engine = LoopGenerationEngine(
            EngineConfig.from_defaults(
                artifact_root=HISTORY_STORE.artifact_root,
                provider_credentials=credentials,
                prompt_override=load_app_prompt_override(),
                default_soundfont_path=selected_soundfont,
            )
        )
        request = GenerationRequest(
            key=key,
            scale=scale,
            description=description,
            model=model_choice,
            temperature=temp,
            use_thinking=use_thinking,
            effort=effort,
            render_audio=True,
            soundfont_path=selected_soundfont,
        )
        progress_events = Queue()

        def handle_progress(event):
            progress_events.put(event)

        # Yield initial status and show the stop-waiting button.
        yield None, None, None, "Working on it...", gr.update(visible=True), None, None, None

        # Run the synchronous Core engine in a background thread so the UI can stop waiting.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                engine.generate,
                request,
                handle_progress,
            )

            # Poll for completion, yielding periodically so Gradio can interrupt the wait.
            while not future.done():
                try:
                    progress_event = progress_events.get(timeout=0.5)
                    status_message = progress_event.message
                except Empty:
                    status_message = "Generating MIDI..."
                yield None, None, None, status_message, gr.update(visible=True), None, None, None

            while not progress_events.empty():
                progress_event = progress_events.get()
                yield (
                    None,
                    None,
                    None,
                    progress_event.message,
                    gr.update(visible=True),
                    None,
                    None,
                    None,
                )

            # Get the result (will raise exception if the API call failed)
            result = future.result()

        print(f"Total cost: {result.cost}")
        visualization = visualize_midi_plotly(MidiFile(result.midi_path))

        # Final yield with the completed result and hide the stop-waiting button.
        yield (
            result.midi_path,
            result.audio_path,
            visualization,
            "",
            gr.update(visible=False),
            result.generation_id,
            result.metadata.soundfont,
            result.audio_path,
        )

    except Exception as e:
        # Catch any exception and yield the error message, hide the stop-waiting button.
        yield None, None, None, str(e), gr.update(visible=False), None, None, None


def toggle_history_sidebar(is_visible):
    """Toggle the visibility of the history sidebar.

    Args:
        is_visible (bool): Current visibility state.

    Returns:
        tuple: (new_visibility, button_text, sidebar_update, history_html, dropdown_update)
    """
    new_visible = not is_visible
    button_text = "Hide History" if new_visible else "History"
    history_html = render_history_html() if new_visible else ""
    choices = get_history_choices() if new_visible else []
    return (
        new_visible,
        button_text,
        gr.update(visible=new_visible),
        history_html,
        gr.update(choices=choices, value=None),
    )


def render_history_html():
    """Render the history items as HTML.

    Returns:
        str: HTML string for the history items.
    """
    history = load_history()

    if not history:
        return """
        <div style="padding: 20px; text-align: center; color: #888;">
            <p>No generations yet.</p>
            <p style="font-size: 0.9em;">Your generated loops will appear here.</p>
        </div>
        """

    html_parts = []
    for gen in history:
        timestamp_str = gen.timestamp.strftime("%b %d, %I:%M %p")
        cost_str = f"${gen.cost:.4f}" if gen.cost is not None else "N/A"
        prompt_preview = gen.prompt[:40] + "..." if len(gen.prompt) > 40 else gen.prompt

        html_parts.append(f"""
        <div class="history-item" data-id="{gen.id}" style="
            background: #2a2a2a;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            border: 1px solid #444;
        ">
            <div style="font-weight: bold; color: #fff; margin-bottom: 4px;">
                {gen.key} {gen.scale}
            </div>
            <div style="font-size: 0.85em; color: #aaa; margin-bottom: 6px;">
                "{prompt_preview}"
            </div>
            <div style="font-size: 0.8em; color: #888; display: flex; justify-content: space-between;">
                <span>{gen.model}</span>
                <span>{timestamp_str}</span>
            </div>
            <div style="font-size: 0.75em; color: #666; margin-top: 4px;">
                Cost: {cost_str}
            </div>
        </div>
        """)

    return "".join(html_parts)


def get_history_choices():
    """Get the history items as choices for the dropdown.

    Returns:
        list: List of (label, value) tuples for dropdown choices.
    """
    history = load_history()
    choices = []
    for gen in history:
        timestamp_str = gen.timestamp.strftime("%b %d %I:%M%p")
        prompt_preview = gen.prompt[:25] + "..." if len(gen.prompt) > 25 else gen.prompt
        label = f"{gen.key} {gen.scale} - {prompt_preview} ({timestamp_str})"
        choices.append((label, gen.id))
    return choices


def load_history_item(gen_id):
    """Load a history item into the main view.

    Args:
        gen_id (str): The generation ID to load.

    Returns:
        tuple: (midi_path, audio_path, soundfont_update, visualization, error_message,
               generation_id, saved_soundfont, current_audio_path, rerender_update)
    """
    if not gen_id:
        return (
            None,
            None,
            get_soundfont_dropdown_update(),
            None,
            "No generation selected",
            None,
            None,
            None,
            get_rerender_button_update(),
        )

    gen = get_generation(gen_id)
    if not gen:
        return (
            None,
            None,
            get_soundfont_dropdown_update(),
            None,
            f"Generation {gen_id} not found",
            None,
            None,
            None,
            get_rerender_button_update(),
        )

    # Check if files exist
    if not os.path.exists(gen.midi_path):
        return (
            None,
            None,
            get_soundfont_dropdown_update(gen.soundfont),
            None,
            f"MIDI file not found: {gen.midi_path}",
            None,
            None,
            None,
            get_rerender_button_update(gen.soundfont, None),
        )

    missing_soundfont_message = ""
    if gen.soundfont:
        saved_soundfont_name = os.path.basename(gen.soundfont)
        if saved_soundfont_name not in get_soundfont_choices():
            missing_soundfont_message = (
                f"Previously used SoundFont: {saved_soundfont_name} (missing)"
            )

    # Load visualization
    try:
        midi = MidiFile(gen.midi_path)
        visualization = visualize_midi_plotly(midi)
    except Exception:
        visualization = None

    # Get audio path if it exists
    audio_path = gen.audio_path if gen.audio_path and os.path.exists(gen.audio_path) else None

    return (
        gen.midi_path,
        audio_path,
        get_soundfont_dropdown_update(gen.soundfont),
        visualization,
        missing_soundfont_message,
        gen.id,
        gen.soundfont,
        audio_path,
        get_rerender_button_update(gen.soundfont, gen.midi_path),
    )


def delete_history_item(
    gen_id,
    current_generation_id=None,
    soundfont_choice=None,
    midi_path=None,
    current_saved_soundfont=None,
    current_audio_path=None,
):
    """Delete a history item.

    Args:
        gen_id (str): The generation ID to delete.

    Returns:
        tuple: (dropdown_update, status_message, history_html, midi_path, audio_path,
               visualization, generation_id, saved_soundfont, current_audio_path,
               rerender_update)
    """
    if not gen_id:
        return (
            gr.update(choices=get_history_choices(), value=None),
            "No generation selected",
            render_history_html(),
            gr.update(),
            gr.update(),
            gr.update(),
            current_generation_id,
            current_saved_soundfont,
            current_audio_path,
            get_rerender_button_update(soundfont_choice, midi_path),
        )

    success = delete_generation(gen_id)
    choices = get_history_choices()
    deleted_active_generation = success and gen_id == current_generation_id
    if success:
        return (
            gr.update(choices=choices, value=None),
            "Deleted generation",
            render_history_html(),
            None if deleted_active_generation else gr.update(),
            None if deleted_active_generation else gr.update(),
            None if deleted_active_generation else gr.update(),
            None if deleted_active_generation else current_generation_id,
            None if deleted_active_generation else current_saved_soundfont,
            None if deleted_active_generation else current_audio_path,
            get_rerender_button_update(
                soundfont_choice,
                None if deleted_active_generation else midi_path,
            ),
        )
    else:
        return (
            gr.update(choices=choices, value=None),
            "Failed to delete generation",
            render_history_html(),
            gr.update(),
            gr.update(),
            gr.update(),
            current_generation_id,
            current_saved_soundfont,
            current_audio_path,
            get_rerender_button_update(soundfont_choice, midi_path),
        )


def refresh_history():
    """Refresh the history display.

    Returns:
        tuple: (dropdown_update, history_html)
    """
    choices = get_history_choices()
    return gr.update(choices=choices, value=None), render_history_html()


def create_demo(playback_status=None):
    """Build and return the Gradio demo."""
    default_soundfont = get_selected_soundfont()
    if playback_status is None:
        playback_status = is_playback_available(default_soundfont)

    playback_available, playback_error = playback_status

    with gr.Blocks(
        css="""
        .center-title { text-align: center; font-size: 3em; }
        .app-header {
            position: relative;
        }
        .app-header .history-toggle {
            position: absolute;
            right: 0;
            top: 50%;
            transform: translateY(-50%);
            z-index: 1;
        }
        .history-sidebar {
            background: #1a1a1a;
            border-left: 1px solid #333;
            height: 100%;
            overflow-y: auto;
        }
        .history-item:hover {
            border-color: #666 !important;
            cursor: pointer;
        }
        """
    ) as demo:
        # State for sidebar visibility
        sidebar_visible = gr.State(value=False)
        current_generation_id = gr.State(value=None)
        current_saved_soundfont = gr.State(value=None)
        current_audio_path = gr.State(value=None)

        # Header with title centered on the original full-width layout
        with gr.Row(elem_classes=["app-header"]):
            gr.Markdown("<h1 class='center-title'>LoopGPT</h1>")
            history_toggle_btn = gr.Button(
                "History",
                size="sm",
                elem_classes=["history-toggle"],
            )

        # Main content area with sidebar
        with gr.Row():
            # Main content column
            with gr.Column(scale=3):
                # Text to MIDI Tab for generating loops based on user input
                with gr.Tab(label="Text to MIDI"):
                    gr.Markdown("Generate a loop based on your description.")
                    with gr.Row():
                        with gr.Accordion("API Keys", open=False):
                            openai_key_input = gr.Textbox(
                                lines=1, type="password", label="OpenAI API Key", value=""
                            )
                            gemini_key_input = gr.Textbox(
                                lines=1, type="password", label="Gemini API Key", value=""
                            )
                            claude_key_input = gr.Textbox(
                                lines=1, type="password", label="Claude API Key", value=""
                            )
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Loop Parameters")
                            key_input = gr.Dropdown(
                                choices=[
                                    "C",
                                    "C#/Db",
                                    "D",
                                    "D#/Eb",
                                    "E",
                                    "F",
                                    "F#/Gb",
                                    "G",
                                    "G#/Ab",
                                    "A",
                                    "A#/Bb",
                                    "B",
                                ],
                                label="Key",
                                value="C",
                            )
                            mode_input = gr.Dropdown(
                                choices=["Major", "minor"], label="Scale", value="Major"
                            )
                            description_input = gr.Textbox(
                                label="Description", value="A rhythmic sad pop song"
                            )
                        with gr.Column():
                            gr.Markdown("## Generation Parameters")
                            default_provider = DEFAULT_PROVIDER
                            default_model = DEFAULT_MODEL
                            default_settings = get_model_settings(
                                default_provider, default_model, False
                            )
                            provider_input = gr.Dropdown(
                                choices=get_providers(), label="Provider", value=default_provider
                            )
                            model_choice_input = gr.Dropdown(
                                choices=get_model_dropdown_choices(default_provider),
                                label="Model",
                                value=default_settings["selected_model"],
                            )
                            temp_input = gr.Slider(
                                0.0,
                                1.0,
                                step=0.1,
                                value=default_settings["temperature_value"],
                                label="Temperature",
                                visible=default_settings["show_temperature"],
                            )
                            thinking_checkbox = gr.Checkbox(
                                label="Reasoning",
                                value=default_settings["thinking_value"],
                                visible=default_settings["show_thinking"],
                            )
                            effort_input = gr.Dropdown(
                                choices=default_settings["effort_options"],
                                label="Reasoning Effort",
                                value=default_settings["effort_value"],
                                visible=default_settings["show_effort"],
                            )
                    with gr.Row():
                        prog_button = gr.Button("Generate Loop", variant="primary")
                        stop_waiting_button = gr.Button(
                            "Stop Waiting", variant="stop", visible=False
                        )

                    # Output section
                    with gr.Row():
                        with gr.Column():
                            prog_output = gr.File(label="Download Generated MIDI")
                            # Audio playback component
                            audio_output = gr.Audio(
                                label="Playback", type="filepath", interactive=False
                            )
                            # Show playback status if not available
                            if not playback_available:
                                gr.Markdown(
                                    f"*{get_soundfont_status_message(default_soundfont)}*",
                                    elem_classes=["warning-text"],
                                )

                    with gr.Row(equal_height=False):
                        soundfont_input = gr.Dropdown(
                            choices=get_soundfont_choices(),
                            label="SoundFont",
                            value=default_soundfont,
                            interactive=True,
                        )
                        with gr.Column():
                            refresh_soundfonts_button = gr.Button("Refresh SoundFonts")
                            rerender_button = gr.Button(
                                "Re-render Audio",
                                interactive=rerender_available(default_soundfont, None),
                            )

                    vis_output = gr.Plot(label="MIDI Visualization")
                    error_message = gr.Textbox(label="Error Message", interactive=False)

                    # Update model choices when provider changes
                    provider_input.change(
                        sync_controls_for_provider,
                        inputs=provider_input,
                        outputs=[
                            model_choice_input,
                            temp_input,
                            thinking_checkbox,
                            effort_input,
                        ],
                    )
                    model_choice_input.change(
                        sync_controls_for_model,
                        inputs=[provider_input, model_choice_input],
                        outputs=[
                            model_choice_input,
                            temp_input,
                            thinking_checkbox,
                            effort_input,
                        ],
                    )
                    thinking_checkbox.change(
                        sync_controls_for_thinking,
                        inputs=[provider_input, model_choice_input, thinking_checkbox],
                        outputs=[
                            model_choice_input,
                            temp_input,
                            thinking_checkbox,
                            effort_input,
                        ],
                    )
                    # When the user clicks the button, run the loop generation function based on the current inputs.
                    # Capture the event so the stop-waiting button can detach the UI from the in-flight request.
                    gen_event = prog_button.click(
                        run_loop,
                        inputs=[
                            key_input,
                            mode_input,
                            description_input,
                            temp_input,
                            model_choice_input,
                            thinking_checkbox,
                            effort_input,
                            soundfont_input,
                            openai_key_input,
                            gemini_key_input,
                            claude_key_input,
                        ],
                        outputs=[
                            prog_output,
                            audio_output,
                            vis_output,
                            error_message,
                            stop_waiting_button,
                            current_generation_id,
                            current_saved_soundfont,
                            current_audio_path,
                        ],
                    )
                    # Stop Waiting detaches the UI from the API response wait and hides itself.
                    stop_waiting_button.click(
                        fn=lambda: (
                            None,
                            None,
                            None,
                            "Stopped waiting. The provider request may still finish in the background.",
                            gr.update(visible=False),
                            None,
                            None,
                            None,
                        ),
                        outputs=[
                            prog_output,
                            audio_output,
                            vis_output,
                            error_message,
                            stop_waiting_button,
                            current_generation_id,
                            current_saved_soundfont,
                            current_audio_path,
                        ],
                        cancels=[gen_event],
                    ).then(
                        get_rerender_button_update,
                        inputs=[soundfont_input, prog_output],
                        outputs=[rerender_button],
                    )
                    rerender_button.click(
                        rerender_current_audio,
                        inputs=[
                            prog_output,
                            soundfont_input,
                            current_saved_soundfont,
                            current_generation_id,
                            current_audio_path,
                        ],
                        outputs=[
                            audio_output,
                            error_message,
                            current_saved_soundfont,
                            current_audio_path,
                        ],
                    )
                    refresh_soundfonts_button.click(
                        refresh_soundfont_controls,
                        inputs=[soundfont_input, prog_output],
                        outputs=[soundfont_input, rerender_button, error_message],
                    )

                # Prompt Editor Tab to allow users to edit the system prompts used in the generation process
                with gr.Tab(label="Prompt Editor"):
                    gr.Markdown("## Edit System Prompt")
                    loop_gen_text = get_prompt_editor_text()
                    # Create text boxes for the user to edit the prompts
                    gr.Markdown("### Loop Generation Prompt")
                    gr.Markdown(
                        "This prompt is used to generate the loop based on the description."
                    )
                    loop_gen_input = gr.Textbox(lines=30, value=loop_gen_text)
                    save_button = gr.Button("Save Prompt")
                    save_status = gr.Textbox(label="Status", interactive=False)
                    # When the user clicks the save button, save the current prompts in the textboxes to the text files
                    save_button.click(
                        save_prompts,
                        inputs=[loop_gen_input],
                        outputs=[save_status],
                    )

            # History sidebar (initially hidden)
            with gr.Column(
                scale=1, visible=False, elem_classes=["history-sidebar"]
            ) as history_sidebar:
                gr.Markdown("## History")

                # Dropdown to select a generation
                history_dropdown = gr.Dropdown(
                    label="Select Generation",
                    choices=get_history_choices(),
                    interactive=True,
                )

                with gr.Row():
                    load_btn = gr.Button("Load", size="sm", variant="primary")
                    delete_btn = gr.Button("Delete", size="sm", variant="stop")
                    refresh_btn = gr.Button("Refresh", size="sm")

                # History items display
                history_html = gr.HTML(
                    value=render_history_html(),
                    label="Recent Generations",
                )

                # Status message for history operations
                history_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False,
                )

        # History sidebar toggle
        history_toggle_btn.click(
            toggle_history_sidebar,
            inputs=[sidebar_visible],
            outputs=[
                sidebar_visible,
                history_toggle_btn,
                history_sidebar,
                history_html,
                history_dropdown,
            ],
        )

        # Load history item into main view
        load_btn.click(
            load_history_item,
            inputs=[history_dropdown],
            outputs=[
                prog_output,
                audio_output,
                soundfont_input,
                vis_output,
                error_message,
                current_generation_id,
                current_saved_soundfont,
                current_audio_path,
                rerender_button,
            ],
        )

        # Delete history item
        delete_btn.click(
            delete_history_item,
            inputs=[
                history_dropdown,
                current_generation_id,
                soundfont_input,
                prog_output,
                current_saved_soundfont,
                current_audio_path,
            ],
            outputs=[
                history_dropdown,
                history_status,
                history_html,
                prog_output,
                audio_output,
                vis_output,
                current_generation_id,
                current_saved_soundfont,
                current_audio_path,
                rerender_button,
            ],
        )

        # Refresh history
        refresh_btn.click(
            refresh_history,
            outputs=[history_dropdown, history_html],
        )

        # Also refresh history after generation completes (when the stop-waiting button becomes hidden)
        # We do this by having the generation flow trigger a refresh
        gen_event.then(
            get_rerender_button_update,
            inputs=[soundfont_input, prog_output],
            outputs=[rerender_button],
        ).then(
            refresh_history,
            outputs=[history_dropdown, history_html],
        )

    return demo


def main():
    """Run the Gradio app."""
    # Surface conductor_core (and app) log records on the console. Core only
    # emits records; configuring handlers is the application's responsibility.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    default_soundfont = get_selected_soundfont()
    playback_status = is_playback_available(default_soundfont)
    playback_available, _ = playback_status
    if not playback_available:
        print(f"Warning: {get_playback_status_message(default_soundfont)}")

    demo = create_demo(playback_status=playback_status)
    demo.launch(allowed_paths=[str(Path(HISTORY_STORE.artifact_root).resolve())])


if __name__ == "__main__":
    main()
