import streamlit as st
import litellm
from dotenv import load_dotenv, find_dotenv
from src.llm_factory import get_valid_llm_list
# , get_llm_instance


@st.cache_resource
def set_langfuse_callbacks():
    _ = load_dotenv(find_dotenv())  # read local .env file
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]  # logs errors to langfuse


def fetch(input_text, selected_model):
    # openai call
    response = litellm.completion(
        model=selected_model,
        messages=[{"role": "user", "content": input_text}],
        # metadata={
        #     "generation_name": "litellm-ishaan-gen",  # set langfuse generation name
        #     # custom metadata fields
        #     "project": "litellm-proxy",
        # },
    )
    return response["choices"][0]["message"]["content"]


set_langfuse_callbacks()

llm_list = get_valid_llm_list()

st.title("LLM Evaluation")
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

input_text = st.text_area("What do you want to say")

final_results = st.container(border=True)
submitted = st.button("Run")
if submitted:
    response = fetch(input_text, selected_model)
    final_results.write(response)
