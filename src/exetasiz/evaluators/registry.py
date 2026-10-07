"""Evaluator registry and factory mechanism."""

from __future__ import annotations

from typing import Any, Callable, Dict, List
from exetasiz.evaluators.base import Evaluator
from exetasiz.evaluators.exact_match import ExactMatchEvaluator
from exetasiz.evaluators.placeholders import (
    HallucinationEvaluator,
    LLMJudgeEvaluator,
    SemanticSimilarityEvaluator,
)

_REGISTRY: Dict[str, Callable[..., Evaluator]] = {}


def register_evaluator(name: str, factory_or_cls: Callable[..., Evaluator]) -> None:
    """Register an evaluator under a unique name."""
    _REGISTRY[name.lower()] = factory_or_cls


def get_evaluator(evaluator_type: str, **kwargs: Any) -> Evaluator:
    """Instantiate an evaluator by name.

    Args:
        evaluator_type: Name of evaluator (e.g., 'exact_match').
        **kwargs: Arguments passed to the evaluator constructor.

    Returns:
        Evaluator instance.

    Raises:
        ValueError: If evaluator name is not found in the registry.
    """
    key = evaluator_type.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown evaluator '{evaluator_type}'. Available evaluators: {available}"
        )
    return _REGISTRY[key](**kwargs)


def list_registered_evaluators() -> List[str]:
    """Return a sorted list of registered evaluator names."""
    return sorted(_REGISTRY.keys())


# Default registrations
register_evaluator("exact_match", ExactMatchEvaluator)
register_evaluator("semantic_similarity", SemanticSimilarityEvaluator)
register_evaluator("llm_judge", LLMJudgeEvaluator)
register_evaluator("hallucination", HallucinationEvaluator)
