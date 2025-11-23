'''
This file is using Gradio for the LoopGPT application. It makes the generation progress more user friendly by providing a GUI for the user to interact with.
'''
from src.midi_processing import loop_to_midi
from src.utils import visualize_midi_beats
import src.ollama_api as ollama_api
import src.runs as runs
from datetime import datetime
from mido import MidiFile
from PIL import Image
import gradio as gr
import io
import os
import json

# Load model list and pricing details from a JSON file
with open('model_list.json', 'r') as f:
    model_info = json.load(f)

def update_temp_visibility(model_choice, use_thinking):
    """This function updates the visibility of the temperature slider based on the selected model and thinking option.

    Args:
        model_choice (str): The selected model choice.
        use_thinking (bool): Whether extended thinking is enabled.

    Returns:
        gr.update(): A Gradio update object to set the visibility of the temperature slider.
    """
    # Hide temperature for o1 models (they don't support temperature)
    if model_choice in model_info["models"]["OpenAI"].keys() and model_info["models"]["OpenAI"][model_choice]["extended_thinking"]:
        return gr.update(visible=False)
    
    # Hide temperature for Claude models when thinking is enabled (temperature must be 1.0)
    elif model_choice in model_info["models"]["Anthropic"].keys() and use_thinking and model_info["models"]["Anthropic"][model_choice]["extended_thinking"]:
        return gr.update(visible=False, value=1.0)

    # Hide temperature for Gemini-3-Pro-Preview (Google recommends removing this parameter and using the Gemini 3 default of 1.0)
    elif model_choice == "gemini-3-pro-preview":
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
    anthropic_thinking = model_choice in model_info["models"]["Anthropic"].keys() and model_info["models"]["Anthropic"][model_choice]["extended_thinking"]
    gemini_thinking = model_choice in model_info["models"]["Google"].keys() and model_info["models"]["Google"][model_choice]["extended_thinking"]
    
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
    openai_reasoning = model_choice in model_info["models"]["OpenAI"].keys() and model_info["models"]["OpenAI"][model_choice]["extended_thinking"]
    if openai_reasoning and model_choice == "gpt-5.1":
        return gr.update(choices=["none", "low", "medium", "high"], value="medium", visible=True)
    elif openai_reasoning and model_choice in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
        return gr.update(choices=["minimal", "low", "medium", "high"], value="medium", visible=True)
    elif openai_reasoning:
        return gr.update(choices=["low", "medium", "high"], value="medium", visible=True)
    else:
        return gr.update(value="medium", visible=False)

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
    return "Prompts saved successfully at " + datetime.now().strftime("%I:%M:%S %p on %B %d, %Y") + "."

def run_loop(key, scale, description, temp, model_choice, use_thinking, effort, translate_prompt_choice, show_visual, openai_key, gemini_key, claude_key):
    """Run the loop generation process based on user inputs and selected model.

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

    Returns:
        str: The path to the generated MIDI file.
        PIL.Image: The MIDI visualization image if show_visual is True, otherwise None.
    """
    error = ""
    try:
        # If the user provided API keys, update environment variables
        if openai_key and openai_key.strip() != "":
            os.environ["OPENAI_API_KEY"] = openai_key.strip()
        if gemini_key and gemini_key.strip() != "":
            os.environ["USER_GEMINI_API_KEY"] = gemini_key.strip()
        if claude_key and claude_key.strip() != "":
            os.environ["ANTHROPIC_API_KEY"] = claude_key.strip()
        
        # Condense the prompt into a single string for the model
        prompt = f"{key} {scale} {description}."
        
        # Generate the loop using the selected model and parameters
        loop, messages, total_cost = runs.generate_midi(
            model_choice=model_choice, 
            prompt=prompt, 
            temp=temp, 
            translate_prompt_choice=translate_prompt_choice, 
            use_thinking=use_thinking,
            effort=effort
        )
        print(f"Total cost: {total_cost}")
        
        # Convert the generated loop into a MIDI file
        midi = MidiFile() 
        loop_to_midi(midi, loop, times_as_string=model_choice in model_info["models"]["Google"].keys())
        output_path = "output.mid"
        midi.save(output_path)
        
        # If the user wants to visualize the MIDI file, generate the visualization
        visualization = None
        if show_visual:
            visualization = Image.open(io.BytesIO(visualize_midi_beats(midi)))
        
        return output_path, visualization, error
    except Exception as e:
        # Catch any exception and return the error message
        error = str(e)
        return None, None, error

# Gradio interface
with gr.Blocks(css=""".center-title { text-align: center; font-size: 3em; }""") as demo:
    gr.Markdown("<h1 class='center-title'>🎶LoopGPT🎶</h1>")
    # Text to MIDI Tab for generating loops based on user input
    with gr.Tab(label="Text to MIDI"):
        gr.Markdown("Generate a loop based on your description.")
        with gr.Row():
            with gr.Accordion("API Keys", open=False):
                openai_key_input = gr.Textbox(lines=1, type="password", label="OpenAI API Key", value="")
                gemini_key_input = gr.Textbox(lines=1, type="password", label="Gemini API Key (Optional)", value="")
                claude_key_input = gr.Textbox(lines=1, type="password", label="Claude API Key", value="")
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Loop Parameters")
                key_input = gr.Dropdown(choices=["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"], label="Key", value="C")
                mode_input = gr.Dropdown(choices=["Major", "minor"], label="Scale", value="Major")
                description_input = gr.Textbox(label="Description", value="A rhythmic sad pop song")
            with gr.Column():
                gr.Markdown("## Generation Parameters")
                model_choice_input = gr.Dropdown(choices=list(model_info["models"]["OpenAI"].keys()) + list(model_info["models"]["Anthropic"].keys()) + list(model_info["models"]["Google"].keys()) + ollama_api.model_list, label="Model", value='gemini-2.5-flash')      
                temp_input = gr.Slider(0.0, 1.0, step=0.1, value=0.1, label="Temperature (t)")
                thinking_checkbox = gr.Checkbox(label="Extended Thinking", value=False, visible=True)
                effort_input = gr.Dropdown(choices=["minimal", "low", "medium", "high"], label="Reasoning Effort", value="medium", visible=False)
                prompt_translate_checkbox = gr.Checkbox(label="Prompt Translation", value=False)
                visualize_checkbox = gr.Checkbox(label="Show MIDI Visualization", value=True)
        prog_button = gr.Button("Generate Loop")
        prog_output = gr.File(label="Download Generated MIDI")  
        vis_output = gr.Image(label="MIDI Visualization")
        error_message = gr.Textbox(label="Error Message", interactive=False)
        # Set visibility of temperature slider and thinking checkbox based on model selection
        model_choice_input.change(
            update_temp_visibility, 
            inputs=[model_choice_input, thinking_checkbox], 
            outputs=temp_input
        )
        model_choice_input.change(
            update_thinking_visibility, 
            inputs=model_choice_input, 
            outputs=thinking_checkbox
        )
        model_choice_input.change(
            update_effort_visibility,
            inputs=model_choice_input,
            outputs=effort_input
        )
        # Also update temperature visibility when thinking checkbox changes
        thinking_checkbox.change(
            update_temp_visibility,
            inputs=[model_choice_input, thinking_checkbox],
            outputs=temp_input
        )
        # When the user clicks the button, run the loop generation function based on the current inputs
        prog_button.click(
            run_loop,
            inputs=[key_input, mode_input, description_input, temp_input, model_choice_input, thinking_checkbox, effort_input, prompt_translate_checkbox, visualize_checkbox, openai_key_input, gemini_key_input, claude_key_input],
            outputs=[prog_output, vis_output, error_message]
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
            gr.Markdown("This prompt is used to generate the loop based on the description.")
            loop_gen_input = gr.Textbox(lines=20, value=loop_gen_text)
            gr.Markdown("### Prompt Translation Prompt")
            gr.Markdown("This prompt is used to translate the description into a more detailed prompt for the model.")
            pt_input = gr.Textbox(lines=20, value=pt_text)
            save_button = gr.Button("Save Prompts")
            save_status = gr.Textbox(label="Status", interactive=False)
            # When the user clicks the save button, save the current prompts in the textboxes to the text files
            save_button.click(save_prompts, inputs=[loop_gen_input, pt_input], outputs=[save_status])

# Launch the demo
demo.launch()