import litellm
from dotenv import load_dotenv, find_dotenv
from response_wrapper import ResponseWrapper


class LlmManager:
    def __init__(self):
        self.set_langfuse_callbacks()

    def set_langfuse_callbacks(self):
        _ = load_dotenv(find_dotenv())  # read local .env file
        # set callbacks
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]  # logs errors to langfuse

    def call_model(self, model_name, messages, tempurature):
        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=tempurature,
            # max_tokens=256,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # metadata={
            #     "generation_name": "litellm-ishaan-gen",  # set langfuse generation name
            #     # custom metadata fields
            #     "project": "litellm-proxy",
            # },
        )
        return ResponseWrapper(response)
