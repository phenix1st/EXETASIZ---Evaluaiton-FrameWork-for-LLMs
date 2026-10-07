"""Example demonstrating how to create and register a custom evaluator in Exetasiz."""

from exetasiz.core.models import TestCase
from exetasiz.evaluators.base import Evaluator, EvaluatorOutput
from exetasiz.evaluators.registry import get_evaluator, register_evaluator


class LengthConstraintEvaluator(Evaluator):
    """Evaluates whether the response length is within a desired word range."""

    def __init__(self, name: str = "length_constraint", max_words: int = 10):
        super().__init__(name=name)
        self.max_words = max_words

    def evaluate(self, test_case: TestCase, generated_answer: str) -> EvaluatorOutput:
        word_count = len(generated_answer.strip().split())
        passed = word_count <= self.max_words
        return EvaluatorOutput(
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"word_count": word_count, "max_allowed": self.max_words},
        )


if __name__ == "__main__":
    # Register evaluator
    register_evaluator("length_constraint", LengthConstraintEvaluator)

    # Instantiate from registry
    evaluator = get_evaluator("length_constraint", max_words=5)

    test_case = TestCase(
        id="demo_01",
        category="conciseness",
        prompt="State your name",
        expected_answer="Exetasiz",
    )

    out_short = evaluator.evaluate(test_case, "I am Exetasiz")
    print("Short answer passed:", out_short.passed, out_short.details)

    out_long = evaluator.evaluate(test_case, "I am an ultra comprehensive intelligent LLM evaluation system built for benchmarking")
    print("Long answer passed:", out_long.passed, out_long.details)
