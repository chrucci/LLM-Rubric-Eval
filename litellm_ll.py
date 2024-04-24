# Low level tracing control
import streamlit as st
import litellm
from dotenv import load_dotenv, find_dotenv
from src.llm_factory import get_valid_llm_list
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

# , get_llm_instance


@st.cache_resource
def set_langfuse_callbacks():
    _ = load_dotenv(find_dotenv())  # read local .env file
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]  # logs errors to langfuse


def call_model(model_name, messages):
    response = litellm.completion(
        model=model_name,
        messages=messages,
        # metadata={
        #     "generation_name": "litellm-ishaan-gen",  # set langfuse generation name
        #     # custom metadata fields
        #     "project": "litellm-proxy",
        # },
    )
    return response["choices"][0]["message"]["content"]


@observe(as_type="generation")
def fetch_with_prompt(input_text, selected_model, selected_prompt):
    prompt = langfuse.get_prompt(selected_prompt)
    messages = prompt.compile(input=input_text)

    langfuse_context.update_current_observation(
        prompt=prompt,
    )
    return call_model(selected_model, messages)


set_langfuse_callbacks()
langfuse = Langfuse()

llm_list = get_valid_llm_list()

st.title("LLM Evaluation (LiteLLM)")
selected_model = st.selectbox(
    "Model",
    [
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        "gpt-4",
        "ollama/codellama",
        "ollama/mistral",
        "ollama/mixtral",
        "ollama/llama2",
        "ollama/llama3",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "gpt-4-turbo",
    ],
)

prompt_options = {
    "Default": "Interact with the LLM without a specfic system prompt.",
    "Analyze Claims": "Parses the body of an article looks for logical claims made.  It then analyzes the validity of those claims.",
}


prompt_summary = st.container()


def prompt_selected(selected_prompt):
    st.write(prompt_options[selected_prompt])


selected_prompt = st.selectbox(
    "Prompts",
    options=prompt_options.keys(),
)
describe_clicked = st.button("Describe Prompt")
if describe_clicked:
    prompt_selected(selected_prompt)

input_text = st.text_area("What do you want to say")

final_results = st.container(border=True)

prompt_submitted = st.button("Prompt")
if prompt_submitted:
    response = fetch_with_prompt(input_text, selected_model, selected_prompt)
    final_results.write(response)
