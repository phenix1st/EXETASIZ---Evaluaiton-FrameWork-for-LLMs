"""Runner for orchestrating evaluations across models, datasets, and evaluators."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from exetasiz.core.models import Dataset, EvalResult, RunResult, TestCase
from exetasiz.evaluators.base import Evaluator
from exetasiz.providers.base import LLMProvider, ProviderResponse


ProgressCallback = Callable[[int, int, str, str], None]


class Runner:
    """Orchestrates running datasets against multiple LLM providers and evaluators."""

    def __init__(
        self,
        dataset: Dataset,
        providers: List[LLMProvider],
        evaluators: List[Evaluator],
        experiment_name: str = "experiment",
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dataset = dataset
        self.providers = providers
        self.evaluators = evaluators
        self.experiment_name = experiment_name
        self.config_snapshot = config_snapshot or {}

    def run(self, progress_callback: Optional[ProgressCallback] = None) -> RunResult:
        """Executes the evaluation suite sequentially.

        Provider errors on any test case are caught and stored without halting execution.

        Args:
            progress_callback: Optional callback invoked after each (provider, test_case) step
                               with signature (current_step, total_steps, provider_name, test_case_id).

        Returns:
            RunResult containing all individual EvalResult records.
        """
        results: List[EvalResult] = []
        total_steps = len(self.providers) * len(self.dataset.test_cases)
        current_step = 0

        for provider in self.providers:
            for test_case in self.dataset.test_cases:
                current_step += 1
                eval_results = self._evaluate_single(provider, test_case)
                results.extend(eval_results)

                if progress_callback is not None:
                    progress_callback(
                        current_step,
                        total_steps,
                        provider.name,
                        test_case.id,
                    )

        timestamp_iso = datetime.now(timezone.utc).isoformat()
        return RunResult(
            experiment_name=self.experiment_name,
            timestamp=timestamp_iso,
            results=results,
            config_snapshot=self.config_snapshot,
        )

    def _evaluate_single(
        self,
        provider: LLMProvider,
        test_case: TestCase,
    ) -> List[EvalResult]:
        """Evaluates a single test case for a given provider across all registered evaluators."""
        response: Optional[ProviderResponse] = None
        error_msg: Optional[str] = None

        try:
            response = provider.generate(test_case.prompt)
        except Exception as e:
            error_msg = f"Provider error: {type(e).__name__}: {e}"

        results_for_case: List[EvalResult] = []

        if error_msg is not None or response is None:
            # Create failed EvalResult for each evaluator
            for evaluator in self.evaluators:
                results_for_case.append(
                    EvalResult(
                        test_case_id=test_case.id,
                        model=provider.name,
                        generated_answer="",
                        score=0.0,
                        passed=False,
                        evaluator=evaluator.name,
                        latency_seconds=0.0,
                        input_tokens=None,
                        output_tokens=None,
                        error=error_msg,
                    )
                )
            return results_for_case

        # Evaluate response with each evaluator
        for evaluator in self.evaluators:
            try:
                eval_out = evaluator.evaluate(test_case, response.text)
                eval_error = None
                score = eval_out.score
                passed = eval_out.passed
            except Exception as e:
                eval_error = f"Evaluator error: {type(e).__name__}: {e}"
                score = 0.0
                passed = False

            results_for_case.append(
                EvalResult(
                    test_case_id=test_case.id,
                    model=provider.name,
                    generated_answer=response.text,
                    score=score,
                    passed=passed,
                    evaluator=evaluator.name,
                    latency_seconds=response.latency_seconds,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    error=eval_error,
                )
            )

        return results_for_case
