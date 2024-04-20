import unittest
from src.llm_factory import get_llm_instance, get_valid_llm_list
from src.available_llms import AvailableLLMs
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic


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

    def test_get_llm_instance_claude3sonnet(self):
        llm = AvailableLLMs.Claude3_Sonnet
        llm_instance = get_llm_instance(llm)
        self.assertEqual(llm_instance.model, "claude-3-sonnet-20240229")
        self.assertIsInstance(llm_instance, Anthropic)

    def test_get_llm_instance_claude3opus(self):
        llm = AvailableLLMs.Claude3_Opus
        llm_instance = get_llm_instance(llm)
        self.assertEqual(llm_instance.model, "claude-3-opus-20240229")
        self.assertIsInstance(llm_instance, Anthropic)

    def test_get_llm_instance_llama3(self):
        llm = AvailableLLMs.Llama3
        llm_instance = get_llm_instance(llm)
        self.assertEqual(llm_instance.model, "llama3")
        self.assertIsInstance(llm_instance, Ollama)

    def test_get_valid_llm_list(self):
        llm_list = get_valid_llm_list()
        expected_llm_list = [
            "GPT4",
            "GPT4_Turbo",
            "GPT35_Turbo",
            "Llama3",
            "Claude3_Opus",
            "Claude3_Sonnet",
        ]
        self.assertListEqual(llm_list, expected_llm_list)

    @unittest.skip("Makes live LLM call, which is not needed on a recurring basis")
    def test_get_llm_instance_returns_valid_llm(self):
        llm = AvailableLLMs.GPT35_Turbo
        llm_instance = get_llm_instance(llm)
        response = llm_instance.complete("hello")
        self.assertIsInstance(response.text, str)


if __name__ == "__main__":
    unittest.main()
