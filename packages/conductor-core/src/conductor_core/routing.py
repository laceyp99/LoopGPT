"""Provider routing for Conductor Core."""

from conductor_core.config import ProviderCredentials
from conductor_core.music import get_model_info
from conductor_core.providers import anthropic as claude_api
from conductor_core.providers import google as gemini_api
from conductor_core.providers import ollama as ollama_api
from conductor_core.providers import openai as openai_api


def generate_midi(
    model_choice,
    prompt,
    temp=0.0,
    use_thinking=False,
    effort="low",
    provider_credentials: ProviderCredentials | None = None,
    system_prompt: str | None = None,
):
    """Generate MIDI loop data by routing a prompt to the selected provider."""
    credentials = provider_credentials or ProviderCredentials()
    model_info = get_model_info()

    ollama_status = ollama_api.get_ollama_status(
        force_refresh=True,
        host_address=credentials.ollama_host,
    )
    ollama_models = ollama_status["models"]

    if model_choice in ollama_models:
        loop, messages, loop_cost = ollama_api.loop_gen(
            prompt,
            model_choice,
            temp=temp,
            host_address=credentials.ollama_host,
            system_prompt=system_prompt,
        )
    elif model_choice in model_info["models"]["OpenAI"]:
        loop, messages, loop_cost = openai_api.loop_gen(
            prompt=prompt,
            model=model_choice,
            temp=temp,
            effort=effort,
            api_key=credentials.openai_api_key,
            system_prompt=system_prompt,
        )
    elif model_choice in model_info["models"]["Google"]:
        loop, messages, loop_cost = gemini_api.loop_gen(
            prompt=prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=effort,
            api_key=credentials.google_api_key,
            system_prompt=system_prompt,
        )
    elif model_choice in model_info["models"]["Anthropic"]:
        loop, messages, loop_cost = claude_api.loop_gen(
            prompt=prompt,
            model=model_choice,
            temp=temp,
            use_thinking=use_thinking,
            effort=effort,
            api_key=credentials.anthropic_api_key,
            system_prompt=system_prompt,
        )
    else:
        if not ollama_status["available"]:
            raise ValueError(
                "Invalid Model Selected. If you intended to use Ollama, it is currently unavailable."
            )
        raise ValueError("Invalid Model Selected")

    return loop, messages, loop_cost
