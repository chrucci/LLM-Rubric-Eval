import os
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from dotenv import load_dotenv, find_dotenv
# from langfuse.openai import openai

_ = load_dotenv(find_dotenv())  # read local .env file

langfuse_callback_handler = LlamaIndexCallbackHandler(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)
Settings.callback_manager = CallbackManager([langfuse_callback_handler])
