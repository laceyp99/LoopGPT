import json
import src.gemini_api as gemini_api
import src.claude_api as claude_api
import src.gpt_api as gpt_api
import src.o_api as o_api

# Load model list and pricing details from a JSON file
with open('model_list.json', 'r') as f:
    model_info = json.load(f)

def generate_midi(model_choice, prompt, temp=0.0, translate_prompt_choice=False, use_thinking=False):
    if model_choice in model_info["models"]["OpenAI"]:
        if model_info["models"]["OpenAI"][model_choice]["extended_thinking"]:
            if translate_prompt_choice:
                prompt_translated, messages, pt_cost = o_api.prompt_gen(prompt, model_choice)
                loop, messages, loop_cost = o_api.loop_gen(prompt_translated, model_choice)
            else:
                loop, messages, loop_cost = o_api.loop_gen(prompt, model_choice)
        else:
            if translate_prompt_choice:
                prompt_translated, messages, pt_cost = gpt_api.prompt_gen(prompt, model_choice, temp)
                loop, messages, loop_cost = gpt_api.loop_gen(prompt_translated, model_choice, temp)
            else:
                loop, messages, loop_cost = gpt_api.loop_gen(prompt, model_choice, temp)

    elif model_choice in model_info["models"]["Google"]:
        if translate_prompt_choice:
            prompt_translated, messages, pt_cost = gemini_api.prompt_gen(prompt, model_choice, temp, use_thinking)
            loop, messages, loop_cost = gemini_api.loop_gen(prompt_translated, model_choice, temp, use_thinking)
        else:
            loop, messages, loop_cost = gemini_api.loop_gen(prompt, model_choice, temp, use_thinking)

    elif model_choice in model_info["models"]["Anthropic"]:
        if translate_prompt_choice:
            prompt_translated, messages, pt_cost = claude_api.prompt_gen(prompt, model_choice, temp, use_thinking)
            loop, messages, loop_cost = claude_api.loop_gen(prompt_translated, model_choice, temp, use_thinking)
        else:
            loop, messages, loop_cost = claude_api.loop_gen(prompt, model_choice, temp, use_thinking)
    else:
        raise ValueError("Invalid Model Selected")
    
    total_cost = pt_cost + loop_cost if translate_prompt_choice else loop_cost
    return loop, messages, total_cost
