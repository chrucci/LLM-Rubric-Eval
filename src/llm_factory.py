from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from .available_llms import AvailableLLMs
import os


#  factory method to create a llama_index llm instance when passed in the name of an llm
def get_llm_instance(llm: AvailableLLMs):
    """
    Returns a llama_index llm instance when passed in the name of an llm.
    """
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    match llm:
        case AvailableLLMs.GPT4:
            return OpenAI(model="gpt-4")
        case AvailableLLMs.GPT4_Turbo:
            return OpenAI(model="gpt-4-turbo")
        case AvailableLLMs.GPT35_Turbo:
            return OpenAI(model="gpt-3.5-turbo")
        case AvailableLLMs.Llama3:
            return Ollama(model="llama3", request_timeout=300.0)
        # case "claude2":
        #     return Anthropic(model="claude-2.1")
        case AvailableLLMs.Claude3_Opus:
            return Anthropic(model="claude-3-opus-20240229", api_key=anthropic_api_key)
        case AvailableLLMs.Claude3_Sonnet:
            return Anthropic(
                model="claude-3-sonnet-20240229", api_key=anthropic_api_key
            )
        case _:
            raise ValueError(f"Invalid llm name: {llm.name}")


def get_valid_llm_list():
    return [llm.name for llm in AvailableLLMs]
