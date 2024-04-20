import unittest
from src.llm_factory import get_llm_instance
from src.available_llms import AvailableLLMs
from llama_index.llms.openai import OpenAI


class TestGetLLMInstance(unittest.TestCase):
    def test_get_llm_instance_gpt4(self):
        llm = AvailableLLMs.GPT4
        llm_instance = get_llm_instance(llm)
        self.assertEqual(llm_instance.model, "gpt-4")
        self.assertIsInstance(llm_instance, OpenAI)

    def test_get_llm_instance_gpt4_turbo(self):
        llm = AvailableLLMs.GPT4_Turbo
        llm_instance = get_llm_instance(llm)
        self.assertEqual(llm_instance.model, "gpt-4-turbo")

        self.assertIsInstance(llm_instance, OpenAI)

    def test_get_llm_instance_gpt35_turbo(self):
        llm = AvailableLLMs.GPT35_Turbo
        llm_instance = get_llm_instance(llm)
        self.assertEqual(llm_instance.model, "gpt-3.5-turbo")
        self.assertIsInstance(llm_instance, OpenAI)

    @unittest.skip("Makes live LLM call, which is not needed on a recurring basis")
    def test_get_llm_instance_returns_valid_llm(self):
        llm = AvailableLLMs.GPT35_Turbo
        llm_instance = get_llm_instance(llm)
        response = llm_instance.complete("hello")
        self.assertIsInstance(response.text, str)


if __name__ == "__main__":
    unittest.main()
