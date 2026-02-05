"""
This file is using Gradio for the LoopGPT application. It makes the generation progress more user friendly by providing a GUI for the user to interact with.

Features:
- Text to MIDI generation with multiple AI providers
- Audio playback of generated MIDI using FluidSynth
- Session history with persistent storage (up to 20 generations)
- Toggleable history sidebar panel
"""

from src.midi_processing import loop_to_midi
from src.utils import visualize_midi_plotly
from src.audio import midi_to_mp3, is_playback_available, get_playback_status_message
from src.history import (
    save_generation,
    load_history,
    get_generation,
    delete_generation,
    get_provider_for_model,
)
import src.ollama_api as ollama_api
import src.runs as runs
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from mido import MidiFile
import gradio as gr
import time
import os
import json

# Load model list and pricing details from a JSON file
with open("model_list.json", "r") as f:
    model_info = json.load(f)


def get_providers():
    """Get list of available providers including Ollama if models are available.

    Returns:
        list: List of provider names.
    """
    providers = list(model_info["models"].keys())
    if ollama_api.model_list:
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
        return ollama_api.model_list
    elif provider in model_info["models"]:
        return list(model_info["models"][provider].keys())
    return []


def update_model_choices(provider):
    """Update the model dropdown choices based on the selected provider.

    Args:
        provider (str): The selected provider.

    Returns:
        gr.update(): A Gradio update object with new choices and default value.
    """
    models = get_models_for_provider(provider)
    default_value = models[0] if models else None
    return gr.update(choices=models, value=default_value)


def update_temp_visibility(model_choice, use_thinking):
    """This function updates the visibility of the temperature slider based on the selected model and thinking option.

    Args:
        model_choice (str): The selected model choice.
        use_thinking (bool): Whether extended thinking is enabled.

    Returns:
        gr.update(): A Gradio update object to set the visibility of the temperature slider.
    """
    # Hide temperature for o1 models (they don't support temperature)
    if (
        model_choice in model_info["models"]["OpenAI"].keys()
        and model_info["models"]["OpenAI"][model_choice]["extended_thinking"]
    ):
        return gr.update(visible=False)

    # Hide temperature for Claude models when thinking is enabled (temperature must be 1.0)
    elif (
        model_choice in model_info["models"]["Anthropic"].keys()
        and use_thinking
        and model_info["models"]["Anthropic"][model_choice]["extended_thinking"]
    ):
        return gr.update(visible=False, value=1.0)

    # Hide temperature for Gemini-3-Pro-Preview (Google recommends removing this parameter and using the Gemini 3 default of 1.0)
    elif (
        model_choice == "gemini-3-pro-preview"
        or model_choice == "gemini-3-flash-preview"
        or model_choice == "claude-opus-4-6"
    ):
        return gr.update(visible=False, value=1.0)

    # Show temperature for all other cases
    return gr.update(visible=True, value=0.1)


def update_thinking_visibility(model_choice):
    """This function updates the visibility of the thinking checkbox based on the selected model.
    Only Claude and Gemini models that support thinking will show the checkbox.

    Args:
        model_choice (str): The selected model choice.

    Returns:
        gr.update(): A Gradio update object to set the visibility of the thinking checkbox.
    """
    # Show thinking toggle only for models that support it
    anthropic_thinking = (
        model_choice in model_info["models"]["Anthropic"].keys()
        and model_info["models"]["Anthropic"][model_choice]["extended_thinking"] and model_choice != "claude-opus-4-6"
    )
    gemini_thinking = (
        model_choice in model_info["models"]["Google"].keys()
        and model_info["models"]["Google"][model_choice]["extended_thinking"]
        and model_choice not in ["gemini-3-pro-preview", "gemini-3-flash-preview"]
    )

    if anthropic_thinking or gemini_thinking:
        return gr.update(visible=True)
    else:
        return gr.update(value=False, visible=False)


def update_effort_visibility(model_choice):
    """This function updates the visibility of the reasoning effort based on the selected model.
    Only OpenAI models that support thinking will show the dropdown that determines the effort.

    Args:
        model_choice (str): The selected model choice.

    Returns:
        gr.update(): A Gradio update object to set the choices, selected value, and visibility of the effort dropdown.
    """
    openai_reasoning = (
        model_choice in model_info["models"]["OpenAI"].keys()
        and model_info["models"]["OpenAI"][model_choice]["extended_thinking"]
    )
    if openai_reasoning and model_choice == "gpt-5.2":
        return gr.update(
            choices=["none", "low", "medium", "high", "xhigh"],
            value="none",
            visible=True,
        )
    elif openai_reasoning and model_choice == "gpt-5.1":
        return gr.update(
            choices=["none", "low", "medium", "high"], value="none", visible=True
        )
    elif openai_reasoning and model_choice in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
        return gr.update(
            choices=["minimal", "low", "medium", "high"], value="minimal", visible=True
        )
    elif openai_reasoning:
        return gr.update(choices=["low", "medium", "high"], value="low", visible=True)
    elif model_choice == "gemini-3-pro-preview":
        return gr.update(choices=["low", "high"], value="low", visible=True)
    elif model_choice == "gemini-3-flash-preview":
        return gr.update(
            choices=["minimal", "low", "medium", "high"], value="minimal", visible=True
        )
    elif model_choice == "claude-opus-4-6":
        return gr.update(choices=["low", "medium", "high", "max"], value="low", visible=True)
    else:
        return gr.update(value="low", visible=False)


def save_prompts(loop_gen_text, pt_text):
    """This function saves any changes to the loop generation and prompt translation prompts to the text files.

    Args:
        loop_gen_text (str): The loop generation prompt text.
        pt_text (str): The prompt translation text.

    Returns:
        str: A message indicating the status of the save operation.
    """
    with open("Prompts/loop gen.txt", "w") as f:
        f.write(loop_gen_text)
    with open("Prompts/prompt translation.txt", "w") as f:
        f.write(pt_text)
    return (
        "Prompts saved successfully at "
        + datetime.now().strftime("%I:%M:%S %p on %B %d, %Y")
        + "."
    )


def run_loop(
    key,
    scale,
    description,
    temp,
    model_choice,
    use_thinking,
    effort,
    translate_prompt_choice,
    show_visual,
    openai_key,
    gemini_key,
    claude_key,
):
    """Run the loop generation process based on user inputs and selected model.

    This is a generator function that yields progress updates, allowing the generation
    to be cancelled by Gradio's cancellation mechanism. The API call runs in a background
    thread while the generator periodically yields status updates.

    Args:
        key (str): The key for the loop that the user selects from the dropdown.
        scale (str): The scale for the loop that the user selects from the Major/minor dropdown.
        description (str): A description of the loop that the user input in the text box.
        temp (float): The sampling temperature for the model that the user selects from the slider.
        model_choice (str): The model that the user selects from the dropdown.
        use_thinking (bool): Whether to enable extended thinking for supported Claude and Gemini models.
        effort (str): The reasoning effort level for supported OpenAI models.
        translate_prompt_choice (bool): Whether to translate the prompt or not during the generation process.
        show_visual (bool): Whether to show the MIDI visualization or not in the UI as a result.
        openai_key (str): The OpenAI API key that the user inputs in the text box.
        gemini_key (str): The Gemini API key that the user inputs in the text box.
        claude_key (str): The Claude API key that the user inputs in the text box.

    Yields:
        tuple: (file_path, audio_path, visualization, status_message, cancel_button_update) -
               intermediate yields show progress and keep cancel visible,
               final yield contains the generated MIDI file, audio, and hides the cancel button.
    """
    try:
        # If the user provided API keys, update environment variables
        if openai_key and openai_key.strip() != "":
            os.environ["OPENAI_API_KEY"] = openai_key.strip()
        if gemini_key and gemini_key.strip() != "":
            os.environ["GEMINI_API_KEY"] = gemini_key.strip()
        if claude_key and claude_key.strip() != "":
            os.environ["ANTHROPIC_API_KEY"] = claude_key.strip()

        # Condense the prompt into a single string for the model
        prompt = f"{key} {scale} {description}."

        # Yield initial status and show cancel button - this is a cancellation checkpoint
        yield None, None, None, "Working on it...", gr.update(visible=True)

        # Run API call in background thread so we can yield periodically for cancellation
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                runs.generate_midi,
                model_choice=model_choice,
                prompt=prompt,
                temp=temp,
                translate_prompt_choice=translate_prompt_choice,
                use_thinking=use_thinking,
                effort=effort,
            )

            # Poll for completion, yielding periodically to allow cancellation
            while not future.done():
                time.sleep(0.5)  # Check every 500ms
                yield None, None, None, "Generating MIDI...", gr.update(visible=True)

            # Get the result (will raise exception if the API call failed)
            loop, messages, total_cost = future.result()

        print(f"Total cost: {total_cost}")

        yield None, None, None, "Processing MIDI...", gr.update(visible=True)
        # Convert the generated loop into a MIDI file
        midi = MidiFile()
        loop_to_midi(
            midi,
            loop,
            times_as_string=model_choice in model_info["models"]["Google"].keys(),
        )
        output_path = "output.mid"
        midi.save(output_path)

        # Render audio from MIDI
        yield None, None, None, "Rendering Audio...", gr.update(visible=True)
        audio_path = midi_to_mp3(output_path)

        # If the user wants to visualize the MIDI file, generate the visualization
        visualization = None
        if show_visual:
            visualization = visualize_midi_plotly(midi)

        # Determine provider for history
        provider = get_provider_for_model(model_choice, model_info)

        # Save to history
        save_generation(
            midi_path=output_path,
            prompt=description,
            key=key,
            scale=scale,
            model=model_choice,
            provider=provider,
            temperature=temp,
            cost=total_cost,
            audio_path=audio_path,
        )

        # Final yield with the completed result and hide cancel button
        yield output_path, audio_path, visualization, "", gr.update(visible=False)

    except Exception as e:
        # Catch any exception and yield the error message, hide cancel button
        yield None, None, None, str(e), gr.update(visible=False)


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
        cost_str = f"${gen.cost:.4f}" if gen.cost else "N/A"
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
        tuple: (midi_path, audio_path, visualization, error_message)
    """
    if not gen_id:
        return None, None, None, "No generation selected"

    gen = get_generation(gen_id)
    if not gen:
        return None, None, None, f"Generation {gen_id} not found"

    # Check if files exist
    if not os.path.exists(gen.midi_path):
        return None, None, None, f"MIDI file not found: {gen.midi_path}"

    # Load visualization
    try:
        midi = MidiFile(gen.midi_path)
        visualization = visualize_midi_plotly(midi)
    except Exception as e:
        visualization = None

    # Get audio path if it exists
    audio_path = (
        gen.audio_path if gen.audio_path and os.path.exists(gen.audio_path) else None
    )

    return gen.midi_path, audio_path, visualization, ""


def delete_history_item(gen_id):
    """Delete a history item.

    Args:
        gen_id (str): The generation ID to delete.

    Returns:
        tuple: (dropdown_update, status_message, history_html)
    """
    if not gen_id:
        return (
            gr.update(choices=get_history_choices(), value=None),
            "No generation selected",
            render_history_html(),
        )

    success = delete_generation(gen_id)
    choices = get_history_choices()
    if success:
        return (
            gr.update(choices=choices, value=None),
            "Deleted generation",
            render_history_html(),
        )
    else:
        return (
            gr.update(choices=choices, value=None),
            "Failed to delete generation",
            render_history_html(),
        )


def refresh_history():
    """Refresh the history display.

    Returns:
        tuple: (dropdown_update, history_html)
    """
    choices = get_history_choices()
    return gr.update(choices=choices, value=None), render_history_html()


# Check playback availability on startup
playback_available, playback_error = is_playback_available()
if not playback_available:
    print(f"Warning: {get_playback_status_message()}")

# Gradio interface
with gr.Blocks(
    css="""
    .center-title { text-align: center; font-size: 3em; }
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

    # Header with title and history toggle
    with gr.Row():
        with gr.Column(scale=20):
            gr.Markdown("<h1 class='center-title'>LoopGPT</h1>")
        with gr.Column(scale=1, min_width=100):
            history_toggle_btn = gr.Button("History", size="sm")

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
                        provider_input = gr.Dropdown(
                            choices=get_providers(), label="Provider", value="Google"
                        )
                        model_choice_input = gr.Dropdown(
                            choices=list(model_info["models"]["Google"].keys()),
                            label="Model",
                            value="gemini-2.5-flash",
                        )
                        temp_input = gr.Slider(
                            0.0, 1.0, step=0.1, value=0.1, label="Temperature"
                        )
                        thinking_checkbox = gr.Checkbox(
                            label="Extended Thinking", value=False, visible=True
                        )
                        effort_input = gr.Dropdown(
                            choices=["minimal", "low", "medium", "high"],
                            label="Reasoning Effort",
                            value="medium",
                            visible=False,
                        )
                        prompt_translate_checkbox = gr.Checkbox(
                            label="Prompt Translation", value=False
                        )
                        visualize_checkbox = gr.Checkbox(
                            label="Show MIDI Visualization", value=True
                        )
                with gr.Row():
                    prog_button = gr.Button("Generate Loop", variant="primary")
                    cancel_button = gr.Button("Cancel", variant="stop", visible=False)

                # Output section
                with gr.Row():
                    with gr.Column():
                        prog_output = gr.File(label="Download Generated MIDI")
                        # Audio playback component
                        audio_output = gr.Audio(label="Playback",type="filepath", interactive=False)
                        # Show playback status if not available
                        if not playback_available:
                            gr.Markdown(
                                f"*Audio playback unavailable. {playback_error}*",
                                elem_classes=["warning-text"],
                            )

                vis_output = gr.Plot(label="MIDI Visualization")
                error_message = gr.Textbox(label="Error Message", interactive=False)

                # Update model choices when provider changes
                provider_input.change(
                    update_model_choices,
                    inputs=provider_input,
                    outputs=model_choice_input,
                )
                # Set visibility of temperature slider and thinking checkbox based on model selection
                model_choice_input.change(
                    update_temp_visibility,
                    inputs=[model_choice_input, thinking_checkbox],
                    outputs=temp_input,
                )
                model_choice_input.change(
                    update_thinking_visibility,
                    inputs=model_choice_input,
                    outputs=thinking_checkbox,
                )
                model_choice_input.change(
                    update_effort_visibility,
                    inputs=model_choice_input,
                    outputs=effort_input,
                )
                # Also update temperature visibility when thinking checkbox changes
                thinking_checkbox.change(
                    update_temp_visibility,
                    inputs=[model_choice_input, thinking_checkbox],
                    outputs=temp_input,
                )
                # When the user clicks the button, run the loop generation function based on the current inputs
                # Capture the event so we can cancel it with the cancel button
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
                        prompt_translate_checkbox,
                        visualize_checkbox,
                        openai_key_input,
                        gemini_key_input,
                        claude_key_input,
                    ],
                    outputs=[
                        prog_output,
                        audio_output,
                        vis_output,
                        error_message,
                        cancel_button,
                    ],
                )
                # Cancel button stops waiting for the API response and hides itself
                cancel_button.click(
                    fn=lambda: (
                        None,
                        None,
                        None,
                        "Generation cancelled.",
                        gr.update(visible=False),
                    ),
                    outputs=[
                        prog_output,
                        audio_output,
                        vis_output,
                        error_message,
                        cancel_button,
                    ],
                    cancels=[gen_event],
                )

            # Prompt Editor Tab to allow users to edit the system prompts used in the generation process
            with gr.Tab(label="Prompt Editor"):
                gr.Markdown("## Edit System Prompts")
                ## Load the existing prompts from the text files in the Prompts folder
                # If the files do not exist, set the text to an empty string
                try:
                    with open("Prompts/loop gen.txt", "r") as lg:
                        loop_gen_text = lg.read()
                except Exception:
                    loop_gen_text = ""
                try:
                    with open("Prompts/prompt translation.txt", "r") as pt:
                        pt_text = pt.read()
                except Exception:
                    pt_text = ""
                # Create text boxes for the user to edit the prompts
                gr.Markdown("### Loop Generation Prompt")
                gr.Markdown(
                    "This prompt is used to generate the loop based on the description."
                )
                loop_gen_input = gr.Textbox(lines=20, value=loop_gen_text)
                gr.Markdown("### Prompt Translation Prompt")
                gr.Markdown(
                    "This prompt is used to translate the description into a more detailed prompt for the model."
                )
                pt_input = gr.Textbox(lines=20, value=pt_text)
                save_button = gr.Button("Save Prompts")
                save_status = gr.Textbox(label="Status", interactive=False)
                # When the user clicks the save button, save the current prompts in the textboxes to the text files
                save_button.click(
                    save_prompts,
                    inputs=[loop_gen_input, pt_input],
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
        outputs=[prog_output, audio_output, vis_output, error_message],
    )

    # Delete history item
    delete_btn.click(
        delete_history_item,
        inputs=[history_dropdown],
        outputs=[history_dropdown, history_status, history_html],
    )

    # Refresh history
    refresh_btn.click(
        refresh_history,
        outputs=[history_dropdown, history_html],
    )

    # Also refresh history after generation completes (when cancel button becomes hidden)
    # We do this by having the generation flow trigger a refresh
    gen_event.then(
        refresh_history,
        outputs=[history_dropdown, history_html],
    )

# Launch the demo
demo.launch()
