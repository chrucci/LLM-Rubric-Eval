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


def call_model(model_name, messages, tempurature):
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
    return response


def get_content(response):
    return response["choices"][0]["message"]["content"]


def get_usage(response):
    usage = response["usage"]
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "total_tokens": usage.total_tokens,
    }


def get_model(response):
    return response["model"]


# TODO: Chris, can you move this to a module or class so this can be called by a DeepEval test?
def fetch_with_prompt(input_text, selected_model, selected_prompt, temperature):
    prompt = langfuse.get_prompt(selected_prompt)
    # st.write(prompt)
    trace = langfuse.trace(
        name="llm-rubric",
        session_id=f"{selected_model}---1",
        version="1.0",
        input=f"{{'input_test': {input_text}}}",
        # metadata={"baz": "bat"},
        # tags=["tag1", "tag2"],
        # release="my fav release",
    )
    # span = trace.span(
    #     name="embedding-search",
    #     metadata={"database": "pinecone"},
    #     input={"query": "This document entails the OKR goals for ACME"},
    # )
    # span.generation(name="query-creation")
    # span.span(name="vector-db-search")
    # span.event(name="db-summary")
    messages = prompt.compile(input=input_text)
    gen = trace.generation(name=selected_prompt, input=messages, version=prompt.version)

    # langfuse_context.update_current_observation(
    #     prompt=prompt,
    # )
    response = call_model(selected_model, messages, temperature)
    model = get_model(response)
    usage = get_usage(response)
    st.write(usage)
    # usage = {"completion_tokens": 497, "prompt_tokens": 1792, "total_tokens": 2289}

    # st.write(usage)

    content = get_content(response)
    trace.update(output=content, input=messages)
    # span.end(output=content)
    gen.end(
        output=content,
        model=model,
        usage=usage,
        model_parameters={"tempurature": temperature},
        prompt=prompt,
    )
    return content


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
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.1)
st.write("Selected value:", temperature)

final_results = st.container(border=True)

prompt_submitted = st.button("Prompt")
if prompt_submitted:
    response = fetch_with_prompt(
        input_text, selected_model, selected_prompt, temperature
    )
    final_results.write(response)
