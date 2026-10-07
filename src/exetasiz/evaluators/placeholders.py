"""Extension point stubs for future evaluators (V2+).

These are planned evaluators preserved as documented stubs.
"""

from __future__ import annotations

from exetasiz.core.models import TestCase
from exetasiz.evaluators.base import Evaluator, EvaluatorOutput


class SemanticSimilarityEvaluator(Evaluator):
    """Evaluates semantic similarity between generated and expected answers using text embeddings."""

    def __init__(self, name: str = "semantic_similarity", threshold: float = 0.85) -> None:
        super().__init__(name=name)
        self.threshold = threshold

    def evaluate(self, test_case: TestCase, generated_answer: str) -> EvaluatorOutput:
        """Evaluate semantic similarity using embeddings."""
        raise NotImplementedError(
            "SemanticSimilarityEvaluator is planned for a future release (V2). "
            "Please use ExactMatchEvaluator in V1."
        )


class LLMJudgeEvaluator(Evaluator):
    """Evaluates answers using an LLM-as-a-judge prompt template and scoring criteria."""

    def __init__(self, name: str = "llm_judge", judge_model: str = "gpt-4o") -> None:
        super().__init__(name=name)
        self.judge_model = judge_model

    def evaluate(self, test_case: TestCase, generated_answer: str) -> EvaluatorOutput:
        """Evaluate answer using LLM-as-a-judge."""
        raise NotImplementedError(
            "LLMJudgeEvaluator is planned for a future release (V2). "
            "Please use ExactMatchEvaluator in V1."
        )


class HallucinationEvaluator(Evaluator):
    """Detects factual hallucinations in generated answers against provided context."""

    def __init__(self, name: str = "hallucination") -> None:
        super().__init__(name=name)

    def evaluate(self, test_case: TestCase, generated_answer: str) -> EvaluatorOutput:
        """Detect hallucinations in generated answer."""
        raise NotImplementedError(
            "HallucinationEvaluator is planned for a future release (V2). "
            "Please use ExactMatchEvaluator in V1."
        )
