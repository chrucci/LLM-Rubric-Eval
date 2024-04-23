import streamlit as st
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.decorators import langfuse_context, observe


@observe(as_type="generation")
def nested_generation(prompt_name):
    prompt = langfuse.get_prompt(prompt_name)

    langfuse_context.update_current_observation(
        prompt=prompt,
    )
    return prompt


@observe()
def main():
    nested_generation()


main()


@st.cache_resource
def set_langfuse_callbacks():
    _ = load_dotenv(find_dotenv())  # read local .env file


# Initialize Langfuse client (prompt management)
langfuse = Langfuse()
set_langfuse_callbacks()

langfuse_handler = CallbackHandler()
st.title("LangChain Eval")
# st.write(langfuse_handler.auth_check())

# st.write(langfuse.auth_check())

langfuse_prompt = langfuse.get_prompt("Extract Questions")
# st.write(langfuse_prompt.prompt)
langchain_prompt = ChatPromptTemplate.from_template(
    langfuse_prompt.get_langchain_prompt()
)


model = ChatOpenAI(
    model=langfuse_prompt.config["model"],
    temperature=str(langfuse_prompt.config["temperature"]),
)

chain = langchain_prompt | model
example_input = {
    "input": "Hartford doesn't have a good waterfront, because every good city has a good waterfront."
}

response = chain.invoke(input=example_input, config={"callbacks": [langfuse_handler]})

st.write(response.content)
