"""Exact match evaluator with case, whitespace, punctuation, and numeric normalization."""

from __future__ import annotations

import re
import string
from typing import Optional
from exetasiz.core.models import TestCase
from exetasiz.evaluators.base import Evaluator, EvaluatorOutput


class ExactMatchEvaluator(Evaluator):
    """Evaluates whether the generated answer matches the expected answer exactly.

    Features:
    - Case-insensitive comparison
    - Whitespace trimming and multi-space collapsing
    - Optional punctuation normalization / stripping
    - Numeric-aware equivalence (e.g. '36', '36.0', and '36 ' match)
    """

    def __init__(
        self,
        name: str = "exact_match",
        normalize_punctuation: bool = True,
        numeric_aware: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.normalize_punctuation = normalize_punctuation
        self.numeric_aware = numeric_aware

    def _try_parse_number(self, text: str) -> Optional[float]:
        """Attempt to parse text as a float, stripping surrounding punctuation and currency symbols."""
        cleaned = text.strip()
        # Remove common currency symbols
        cleaned = re.sub(r"^[\$€£¥₹]+", "", cleaned)
        cleaned = re.sub(r"[\$€£¥₹%]+$", "", cleaned).strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _normalize_text(self, text: str) -> str:
        """Normalize string by lowercasing, collapsing whitespaces, and optionally stripping punctuation."""
        normalized = text.strip().lower()
        if self.normalize_punctuation:
            # Replace punctuation with whitespace, then collapse whitespace
            translator = str.maketrans({p: " " for p in string.punctuation})
            normalized = normalized.translate(translator)
        # Collapse multiple spaces into one
        normalized = " ".join(normalized.split())
        return normalized

    def evaluate(self, test_case: TestCase, generated_answer: str) -> EvaluatorOutput:
        expected = test_case.expected_answer
        actual = generated_answer

        # 1. Numeric-aware check if enabled
        if self.numeric_aware:
            expected_num = self._try_parse_number(expected)
            actual_num = self._try_parse_number(actual)
            if expected_num is not None and actual_num is not None:
                matches = abs(expected_num - actual_num) < 1e-6
                return EvaluatorOutput(
                    score=1.0 if matches else 0.0,
                    passed=matches,
                    details={
                        "comparison_type": "numeric",
                        "expected_num": expected_num,
                        "actual_num": actual_num,
                    },
                )

        # 2. Text comparison
        norm_expected = self._normalize_text(expected)
        norm_actual = self._normalize_text(actual)
        matches = norm_expected == norm_actual

        return EvaluatorOutput(
            score=1.0 if matches else 0.0,
            passed=matches,
            details={
                "comparison_type": "text",
                "normalized_expected": norm_expected,
                "normalized_actual": norm_actual,
            },
        )
