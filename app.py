'''
This file is using Gradio for the LoopGPT application. It makes the generation progress more user friendly by providing a GUI for the user to interact with.
'''
from src.midi_processing import loop_to_midi
from src.utils import visualize_midi_beats
import src.gemini_api as gemini_api
import src.claude_api as claude_api
import src.gpt_api as gpt_api
import src.o_api as o_api
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
    if model_choice in model_info["models"]["Anthropic"].keys() and use_thinking and model_info["models"]["Anthropic"][model_choice]["extended_thinking"]:
        return gr.update(visible=False)
    
    # Show temperature for all other cases
    return gr.update(visible=True)

def update_thinking_visibility(model_choice):
    """This function updates the visibility of the thinking checkbox based on the selected model.
    Only Claude models that support thinking will show the checkbox.

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

def run_loop(key, scale, description, temp, model_choice, use_thinking, translate_prompt_choice, show_visual, openai_key, gemini_key, claude_key):
    """Run the loop generation process based on user inputs and selected model.

    Args:
        key (str): The key for the loop that the user selects from the dropdown.
        scale (str): The scale for the loop that the user selects from the Major/minor dropdown.
        description (str): A description of the loop that the user input in the text box.
        temp (float): The sampling temperature for the model that the user selects from the slider.
        model_choice (str): The model that the user selects from the dropdown.
        use_thinking (bool): Whether to enable extended thinking for supported Claude models.
        translate_prompt_choice (bool): Whether to translate the prompt or not during the generation process.
        show_visual (bool): Whether to show the MIDI visualization or not in the UI as a result.
        openai_key (str): The OpenAI API key that the user inputs in the text box.
        gemini_key (str): The Gemini API key that the user inputs in the text box.
        claude_key (str): The Claude API key that the user inputs in the text box.

    Returns:
        str: The path to the generated MIDI file.
        PIL.Image: The MIDI visualization image if show_visual is True, otherwise None.
    """
    # If the user provided API keys, update environment variables
    if openai_key and openai_key.strip() != "":
        os.environ["OPENAI_API_KEY"] = openai_key.strip()
    if gemini_key and gemini_key.strip() != "":
        os.environ["USER_GEMINI_API_KEY"] = gemini_key.strip()
    if claude_key and claude_key.strip() != "":
        os.environ["ANTHROPIC_API_KEY"] = claude_key.strip()
    
    # Condense the prompt into a single string for the model
    prompt = f"{key} {scale} {description}."
    
    gemini_model = False
    pt_cost = 0
    loop_cost = 0
    
    # Route the generation process based on model choice
    
    if model_choice in model_info["models"]["OpenAI"].keys():
        if model_info["models"]["OpenAI"][model_choice]["extended_thinking"]:
            if translate_prompt_choice:
                prompt, messages, pt_cost = o_api.prompt_gen(prompt, model_choice)
            loop, messages, loop_cost = o_api.loop_gen(prompt, model_choice)
        else:
            if translate_prompt_choice:
                prompt, messages, pt_cost = gpt_api.prompt_gen(prompt, model_choice, temp)    
            loop, messages, loop_cost = gpt_api.loop_gen(prompt, model_choice, temp)
    elif model_choice in model_info["models"]["Google"].keys():
        gemini_model = True
        if translate_prompt_choice:
            prompt, messages, pt_cost = gemini_api.prompt_gen(prompt, model_choice, temp, use_thinking)
        loop, messages, loop_cost = gemini_api.loop_gen(prompt, model_choice, temp, use_thinking)
    elif model_choice in model_info["models"]["Anthropic"].keys(): 
        if translate_prompt_choice:
            prompt, messages, pt_cost = claude_api.prompt_gen(prompt, model_choice, temp, use_thinking)    
        loop, messages, loop_cost = claude_api.loop_gen(prompt, model_choice, temp, use_thinking)
    else:
        return "Invalid Model Selected", 0
    
    # Calculate total cost
    total_cost = pt_cost + loop_cost
    print(f"Total cost: {total_cost}")
    
    # Convert the generated loop into a MIDI file
    midi = MidiFile() 
    loop_to_midi(midi, loop, times_as_string=gemini_model)
    output_path = "output.mid"
    midi.save(output_path)
    
    # If the user wants to visualize the MIDI file, generate the visualization
    visualization = None
    if show_visual:
        visualization = Image.open(io.BytesIO(visualize_midi_beats(midi)))
    
    return output_path, visualization

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
                model_choice_input = gr.Dropdown(choices=list(model_info["models"]["OpenAI"].keys()) + list(model_info["models"]["Anthropic"].keys()) + list(model_info["models"]["Google"].keys()), label="Model", value='gemini-2.5-flash')      
                temp_input = gr.Slider(0.0, 1.0, step=0.1, value=0.1, label="Temperature (t)")
                thinking_checkbox = gr.Checkbox(label="Extended Thinking", value=False, visible=True)
                prompt_translate_checkbox = gr.Checkbox(label="Prompt Translation", value=False)
                visualize_checkbox = gr.Checkbox(label="Show MIDI Visualization", value=True)
        prog_button = gr.Button("Generate Loop")
        prog_output = gr.File(label="Download Generated MIDI")  
        vis_output = gr.Image(label="MIDI Visualization")
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
        # Also update temperature visibility when thinking checkbox changes
        thinking_checkbox.change(
            update_temp_visibility,
            inputs=[model_choice_input, thinking_checkbox],
            outputs=temp_input
        )
        # When the user clicks the button, run the loop generation function based on the current inputs
        prog_button.click(
            run_loop,
            inputs=[key_input, mode_input, description_input, temp_input, model_choice_input, thinking_checkbox, prompt_translate_checkbox, visualize_checkbox, openai_key_input, gemini_key_input, claude_key_input],
            outputs=[prog_output, vis_output]
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