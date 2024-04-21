import os
import streamlit as st
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import set_global_handler
from dotenv import load_dotenv, find_dotenv
from src.llm_factory import get_valid_llm_list, get_llm_instance
from src.available_llms import AvailableLLMs

# from langfuse.openai import openai


@st.cache_resource
def get_langfuse_callback_handler():
    _ = load_dotenv(find_dotenv())  # read local .env file

    lf_public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    lf_secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    lf_host = os.environ["LANGFUSE_HOST"]

    langfuse_callback_handler = LlamaIndexCallbackHandler(
        public_key=lf_public_key,
        secret_key=lf_secret_key,
        host=lf_host,
    )
    Settings.callback_manager = CallbackManager([langfuse_callback_handler])

    set_global_handler(
        "langfuse", secret_key=lf_secret_key, public_key=lf_public_key, host=lf_host
    )
    st.write(lf_host)
    st.write(lf_public_key)
    st.write(lf_secret_key)
    return langfuse_callback_handler


def fetch(input_text, selected_model):
    llm = get_llm_instance(AvailableLLMs[selected_model])
    return llm.chat(input_text)


langfuse_callback_handler = get_langfuse_callback_handler()
llm_list = get_valid_llm_list()

st.title("LLM Evaluation")
selected_model = st.selectbox(
    "Model",
    llm_list,
)

input_text = st.text_area("Input text")

submitted = st.button("Run")


final_results = st.container(border=True)
# final_results.write(response)
if submitted:
    response = fetch(input_text, selected_model)
    final_results.write(response)
    langfuse_callback_handler.flush()
