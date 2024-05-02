# This will be a streamlit app just like the others, but all of the LLM calls are referenced from a module in a separate file.

import streamlit as st
from llm_manager import LlmManager
from langfuse import Langfuse
# import os

# # Get the environment variables
# env_vars = os.environ

# # Print the environment variables
# for key, value in env_vars.items():
#     st.write(f"{key}: {value}")

langfuse = Langfuse()


@st.cache_resource
def initialize():
    st.session_state.prompt_text = ""
    return LlmManager()


llmManager = initialize()

st.title("LLM Evaluation (LiteLLM - Mod)")
st.sidebar.header("Options")
selected_model = st.sidebar.selectbox(
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
    ],
)
prompt_options = {
    "Default": "Interact with the LLM without a specfic system prompt.",
    "Analyze Claims": "Parses the body of an article looks for logical claims made.  It then analyzes the validity of those claims.",
    "ExtractWisdom": "Extracts the wisdom from a text.",
    "ExtractWisdom2": "Extracts the wisdom from a text.",
}


def prompt_selected():
    # prompt_desc = prompt_options[st.session_state.selected_prompt]
    # prompt = langfuse.get_prompt(st.session_state.selected_prompt).prompt
    # st.write(prompt)
    prompt_body.write("prompt")


prompt_body = st.container(border=True)
selected_prompt = st.sidebar.selectbox(
    "Prompts",
    options=prompt_options.keys(),
    on_change=prompt_selected,
    key="selected_prompt",
)
# describe_clicked = st.button("Describe Prompt")
# if describe_clicked:
#     prompt_selected()

# input_text = st.text_area(
#     "System Prompt", value=st.session_state.prompt_text, key="prompt_text"
# )

# if st.button("Fill with sample text"):
#     st.session_state.prompt_text = "This is some sample text."


# Replace the placeholder with some text:
# prompt_body.text("Hello")

# # Replace the text with a chart:
# placeholder.line_chart({"data": [1, 5, 2, 6]})

# # Replace the chart with several elements:
# with placeholder.container():
#     st.write("This is one element")
#     st.write("This is another")
