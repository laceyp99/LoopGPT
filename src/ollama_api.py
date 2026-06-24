import ollama
import src.utils as utils
import src.objects as objects
import logging
import sys
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_ollama_status_cache = None

def initialize_ollama_client(host_address="http://localhost:11434"):
    """Initializes and returns an Ollama client.

    Args:
        host_address (str, optional): The host address for the Ollama API. Defaults to "http://localhost:11434", assuming the API is running locally.

    Returns:
        ollama.Client: The initialized Ollama client.
    """
    load_dotenv(os.path.join('src', '.env'))
    if os.getenv('OLLAMA_API_HOST_ADDRESS'):
        client = ollama.Client(
            host=os.getenv('OLLAMA_API_HOST_ADDRESS')
        )
    else:
        client = ollama.Client(
            host=host_address
        )
    return client

def get_ollama_status(force_refresh=False):
    """Get the current Ollama availability and discovered models.

    Args:
        force_refresh (bool, optional): Whether to bypass the cached status. Defaults to False.

    Returns:
        dict: Availability details with keys: available, models, host, error.
    """
    global _ollama_status_cache

    if _ollama_status_cache is not None and not force_refresh:
        return _ollama_status_cache

    load_dotenv(os.path.join('src', '.env'))
    host = os.getenv('OLLAMA_API_HOST_ADDRESS') or "http://localhost:11434"
    status = {
        "available": False,
        "models": [],
        "host": host,
        "error": None,
    }

    try:
        client = initialize_ollama_client(host_address=host)
        status["models"] = [model.model for model in client.list().models]
        status["available"] = True
    except Exception as exc:
        status["error"] = str(exc)
        logger.warning("Ollama unavailable at %s: %s", host, exc)

    _ollama_status_cache = status
    return status


def get_model_list(force_refresh=False):
    """Get the available Ollama models.

    Args:
        force_refresh (bool, optional): Whether to bypass the cached status. Defaults to False.

    Returns:
        list: Available Ollama model names.
    """
    return get_ollama_status(force_refresh=force_refresh)["models"]


def loop_gen(prompt, model, temp=0.0):
    """
    Generate a MIDI bar (chord progression/melody) using the specified model and prompt.

    Args:
        prompt (str): The user prompt to generate MIDI data.
        model (str): The model identifier to use.
        temp (float, optional): Temperature for generation. Defaults to 0.0.

    Returns:
        tuple: (midi_loop, messages, cost=0 for Ollama)
    """
    # Initialize Ollama client and build messages for the API call
    client = initialize_ollama_client()
    loop_prompt = utils.get_loop_prompt()
    messages = [
        {"role": "system", "content": loop_prompt},
        {"role": "user", "content": prompt},
    ]
    # Make the structured output API call for MIDI data generation
    completion = client.chat(
        model=model,
        messages=messages,
        format=objects.Loop.model_json_schema(),
        options={
            "temperature": temp
        }
    )
    # Extract the generated MIDI loop
    message = getattr(completion, "message", None)
    content = getattr(message, "content", None)
    if not content:
        raise ValueError("Ollama response did not include generated content.")

    midi_loop = objects.Loop.model_validate_json(content)
    thinking = getattr(message, "thinking", None)
    if thinking:
        messages.append({"role": "assistant", "content": thinking})
    messages.append({"role": "assistant", "content": str(midi_loop)})
    return midi_loop, messages, 0
