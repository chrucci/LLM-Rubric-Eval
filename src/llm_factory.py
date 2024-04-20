from llama_index.llms.openai import OpenAI
from .available_llms import AvailableLLMs


#  factory method to create a llama_index llm instance when passed in the name of an llm
def get_llm_instance(llm: AvailableLLMs):
    """
    Returns a llama_index llm instance when passed in the name of an llm.
    """
    match llm:
        case AvailableLLMs.GPT4:
            return OpenAI(model="gpt-4")
        case AvailableLLMs.GPT4_Turbo:
            return OpenAI(model="gpt-4-turbo")
        case AvailableLLMs.GPT35_Turbo:
            return OpenAI(model="gpt-3.5-turbo")
        # case "llama2" | "llama3" | "mistrel" | "mixtrel":
        #     return Ollama(model=llm_name, request_timeout=300.0)
        # case "claude2":
        #     return Anthropic(model="claude-2.1")
        # case "claude3":
        #     return Anthropic(model="claude-3-opus-20240229")
        case _:
            raise ValueError(f"Invalid llm name: {llm.name}")


def get_valid_llm_list():
    return ["openai", "llama2", "llama3", "mistrel", "mixtrel", "claude2", "claude3"]
