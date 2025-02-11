'''
This file is using Gradio for the LoopGPT application. It makes the generation progress more user friendly by providing a GUI for the user to interact with.
'''

# IMPORTS
import gradio as gr
import tempfile
import os
import io
from PIL import Image
from mido import MidiFile
import ProgressionPlus
import Accompaniment
import code.utils as utils

def get_temp_midi_filename():
    """Create and return a temporary filename with a .mid suffix."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        return tmp.name

def run_progression(key, mode, description, temp, include_melody, visualize=True):
    """
    Call ProgressionPlus.main to generate a chord progression MIDI file.
    A temporary filename is used to store the generated MIDI.
    """
    # Initialize the generation MIDI file
    output_path = get_temp_midi_filename()
    prompt_dict = {
        "key": key,
        "mode": mode,
        "keywords": description, 
        "t": temp
    }
    ProgressionPlus.main(prompt_dict=prompt_dict, melody=include_melody, play=False, filename=output_path)
    # Determine if the user wants to visualize the MIDI
    image_data = None
    if visualize:
        # Call our new utility to generate visualization image
        img_bytes = utils.visualize_midi(output_path)
        image_data = Image.open(io.BytesIO(img_bytes))
    return output_path, image_data

def run_accompaniment(midi_file, temp, visualize=True):
    """
    Save the uploaded MIDI file to a temporary file, load it as a MidiFile, then invoke Accompaniment.main to generate an accompaniment. 
    A temporary filename is used to store the generated MIDI.
    """
    # If midi_file is received as a file path (string) use it, otherwise write the uploaded file.
    if isinstance(midi_file, dict):
        input_path = midi_file.get("name", "")
    elif isinstance(midi_file, str) and os.path.exists(midi_file):
        input_path = midi_file
    else:
        with open(get_temp_midi_filename(), "wb") as tmp_file:
            tmp_file.write(midi_file.read())
            input_path = tmp_file.name
    # Load the MIDI file
    midi = MidiFile(input_path)
    # Initialize the generation MIDI file
    output_path = get_temp_midi_filename()
    Accompaniment.main(melody=midi, play=False, filename=output_path, temp=temp)
    # Determine if the user wants to visualize the MIDI
    image_data = None
    if visualize:
        img_bytes = utils.visualize_midi(output_path)
        image_data = Image.open(io.BytesIO(img_bytes))
    return output_path, image_data

# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LoopGPT")
    # ProgressionPlus tab
    with gr.Tab("ProgressionPlus"):
        gr.Markdown("Generate a chord progression (with an optional melody) based on your description.")
        key_input = gr.Dropdown(choices=["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"], label="Key", value="C")
        mode_input = gr.Dropdown(choices=["Major", "minor"], label="Mode", value="Major")
        description_input = gr.Textbox(label="Description", value="A rhythmic sad pop song")
        temp_input = gr.Slider(0.0, 1.0, step=0.1, value=0.3, label="Temperature (t)")
        include_melody_input = gr.Checkbox(label="Include Melody", value=False)
        visualize = gr.Checkbox(label="Visualize Generated MIDI", value=True)
        prog_button = gr.Button("Generate Progression")
        prog_output = gr.File(label="Download Generated MIDI")
        prog_image_output = gr.Image(label="Visualization", type="pil")
        # When the button is clicked, the progressionplus main function is called with the user inputs
        prog_button.click(
            run_progression,
            inputs=[key_input, mode_input, description_input, temp_input, include_melody_input, visualize],
            outputs=[prog_output, prog_image_output]
        )
    # Accompaniment tab
    with gr.Tab("Accompaniment"):
        gr.Markdown("Generate an accompaniment based on an uploaded MIDI melody.")
        midi_file_input = gr.File(label="Upload MIDI File", type="filepath")
        temp_acc_input = gr.Slider(0.0, 1.0, step=0.1, value=0.3, label="Temperature")
        visualize = gr.Checkbox(label="Visualize Generated MIDI", value=True)
        acc_button = gr.Button("Generate Accompaniment")
        acc_output = gr.File(label="Download Generated MIDI")
        acc_image_output = gr.Image(label="Visualization", type="pil")
        # When the button is clicked, the accompaniment main function is called with the user inputs
        acc_button.click(
            run_accompaniment,
            inputs=[midi_file_input, temp_acc_input, visualize],
            outputs=[acc_output, acc_image_output]
        )
# Launch the demo
demo.launch()