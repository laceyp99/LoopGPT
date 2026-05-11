import src.ollama_api as ollama_api
import src.gemini_api as gemini_api
import src.claude_api as claude_api
import src.openai_api as openai_api
from src.utils import get_model_info


def generate_midi(
    model_choice,
    prompt,
    temp=0.0,
    use_thinking=False,
    effort="low",
):
    """Generate MIDI loops based on user prompts using the specified model.

    Args:
        model_choice (str): The model to use for generation.
        prompt (str): The user prompt to generate MIDI data.
        temp (float, optional): The temperature for generation. Defaults to 0.0.
        use_thinking (bool, optional): Whether to use extended thinking. Defaults to False.
        effort (str, optional): Reasoning effort level. Defaults to "low".

    Raises:
        ValueError: If an invalid model is selected.

    Returns:
        tuple: (midi_loop, messages, total_cost)
    """
    model_info = get_model_info()
    ollama_status = ollama_api.get_ollama_status(force_refresh=True)
    ollama_models = ollama_status["models"]

    # Ollama models
    if model_choice in ollama_models:
        loop, messages, loop_cost = ollama_api.loop_gen(prompt, model_choice)
    # OpenAI (unified - handles both standard and reasoning models)
    elif model_choice in model_info["models"]["OpenAI"]:
        loop, messages, loop_cost = openai_api.loop_gen(
            prompt=prompt, model=model_choice, temp=temp, effort=effort
        )
    # Google Gemini
    elif model_choice in model_info["models"]["Google"]:
        loop, messages, loop_cost = gemini_api.loop_gen(
            prompt=prompt, model=model_choice, temp=temp, use_thinking=use_thinking, effort=effort
        )
    # Anthropic Claude
    elif model_choice in model_info["models"]["Anthropic"]:
        loop, messages, loop_cost = claude_api.loop_gen(
            prompt=prompt, model=model_choice, temp=temp, use_thinking=use_thinking, effort=effort
        )
    else:
        if not ollama_status["available"]:
            raise ValueError(
                "Invalid Model Selected. If you intended to use Ollama, it is currently unavailable."
            )
        raise ValueError("Invalid Model Selected")

    return loop, messages, loop_cost