import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

models = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    # "gpt-4",
    # "ollama/codellama",
    # "ollama/mistral",
    # "ollama/mixtral",
    # "ollama/llama2",
    # "ollama/llama3",
    # "claude-3-opus-20240229",
    # "claude-3-sonnet-20240229",
]


@pytest.mark.parametrize("model", models)
def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.  But then you'll have to say sorry to everyone you see for a day.",
    )
    assert_test(test_case, [answer_relevancy_metric])


@pytest.mark.parametrize("model", models)
def test_logic__john_mark():
    input = """
 John and Mark are in a room with a ball, a basket and a box.  John puts the ball in the box, then leaves for work.  While John is away, Mark puts the ball in the basket, and then leaves for school. They both come back together later in the day, and they do not know what happened in the room after each of them left the room.  Where do they think the ball is?   
    """
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input=input,
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.  But then you'll have to say sorry to everyone you see for a day.",
    )
    assert_test(test_case, [answer_relevancy_metric])
