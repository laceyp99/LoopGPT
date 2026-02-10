import json
import src.ollama_api as ollama_api
import src.gemini_api as gemini_api
import src.claude_api as claude_api
import src.openai_api as openai_api

# Load model list and pricing details from a JSON file
with open("model_list.json", "r") as f:
    model_info = json.load(f)


def generate_midi(
    model_choice,
    prompt,
    temp=0.0,
    translate_prompt_choice=False,
    use_thinking=False,
    effort="low",
):
    """Generate MIDI loops based on user prompts using the specified model. This function mainly handles routing to the appropriate APIs and then managing the prompt translation step if needed.

    Args:
        model_choice (str): The model to use for generation.
        prompt (str): The user prompt to generate MIDI data.
        temp (float, optional): The temperature for generation. Defaults to 0.0.
        translate_prompt_choice (bool, optional): Whether to translate the prompt. Defaults to False.
        use_thinking (bool, optional): Whether to use extended thinking. Defaults to False.
        effort (str, optional): Reasoning effort level. Defaults to "low".

    Raises:
        ValueError: If an invalid model is selected.

    Returns:
        tuple: (midi_loop, messages, total_cost)
    """
    # Ollama models
    if model_choice in ollama_api.model_list:
        if translate_prompt_choice:
            prompt_translated, messages, pt_cost = ollama_api.prompt_gen(
                prompt, model_choice
            )
            loop, messages, loop_cost = ollama_api.loop_gen(
                prompt_translated, model_choice
            )
        else:
            loop, messages, loop_cost = ollama_api.loop_gen(prompt, model_choice)
    # OpenAI (unified - handles both standard and reasoning models)
    elif model_choice in model_info["models"]["OpenAI"]:
        if translate_prompt_choice:
            prompt_translated, messages, pt_cost = openai_api.prompt_gen(
                prompt, model_choice, temp, effort
            )
            loop, messages, loop_cost = openai_api.loop_gen(
                prompt_translated, model_choice, temp, effort
            )
        else:
            loop, messages, loop_cost = openai_api.loop_gen(
                prompt, model_choice, temp, effort
            )
    # Google Gemini
    elif model_choice in model_info["models"]["Google"]:
        if translate_prompt_choice:
            prompt_translated, messages, pt_cost = gemini_api.prompt_gen(
                prompt, model_choice, temp, use_thinking
            )
            loop, messages, loop_cost = gemini_api.loop_gen(
                prompt_translated, model_choice, temp, use_thinking
            )
        else:
            loop, messages, loop_cost = gemini_api.loop_gen(
                prompt, model_choice, temp, use_thinking
            )
    # Anthropic Claude
    elif model_choice in model_info["models"]["Anthropic"]:
        if translate_prompt_choice:
            prompt_translated, messages, pt_cost = claude_api.prompt_gen(
                prompt, model_choice, temp, use_thinking, effort
            )
            loop, messages, loop_cost = claude_api.loop_gen(
                prompt_translated, model_choice, temp, use_thinking, effort
            )
        else:
            loop, messages, loop_cost = claude_api.loop_gen(
                prompt, model_choice, temp, use_thinking, effort
            )
    else:
        raise ValueError("Invalid Model Selected")

    # Calculate total cost
    total_cost = pt_cost + loop_cost if translate_prompt_choice else loop_cost
    return loop, messages, total_cost
