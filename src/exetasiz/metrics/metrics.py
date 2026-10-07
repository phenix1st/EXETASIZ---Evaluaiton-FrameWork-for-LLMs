"""Metrics calculation and model comparison functions using pandas."""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import pandas as pd
from exetasiz.core.models import AggregateMetric, EvalResult, RunResult, StructuredMetrics, TestCase


def results_to_dataframe(
    run_result_or_evals: Union[RunResult, List[EvalResult]],
    test_cases: Optional[List[TestCase]] = None,
) -> pd.DataFrame:
    """Convert a RunResult or list of EvalResults to a pandas DataFrame."""
    evals = (
        run_result_or_evals.results
        if isinstance(run_result_or_evals, RunResult)
        else run_result_or_evals
    )

    category_map: Dict[str, str] = {}
    if test_cases:
        for tc in test_cases:
            category_map[tc.id] = tc.category

    rows = []
    for r in evals:
        rows.append({
            "test_case_id": r.test_case_id,
            "category": category_map.get(r.test_case_id, "unknown"),
            "model": r.model,
            "evaluator": r.evaluator,
            "score": r.score,
            "passed": r.passed,
            "latency_seconds": r.latency_seconds,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "has_error": r.error is not None,
            "error": r.error,
            "generated_answer": r.generated_answer,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "test_case_id", "category", "model", "evaluator", "score",
            "passed", "latency_seconds", "input_tokens", "output_tokens",
            "has_error", "error", "generated_answer"
        ])
    return df


def compute_structured_metrics(
    eval_results: List[EvalResult],
    test_cases: List[TestCase],
) -> StructuredMetrics:
    """Compute hierarchical StructuredMetrics (overall, by_model, by_category, by_model_and_category)."""
    df = results_to_dataframe(eval_results, test_cases=test_cases)

    def _to_metric(subset: pd.DataFrame) -> AggregateMetric:
        total = len(subset)
        if total == 0:
            return AggregateMetric(
                total_evaluations=0,
                accuracy=0.0,
                pass_count=0,
                fail_count=0,
                error_count=0,
                failure_rate=0.0,
                avg_latency_s=None,
                p95_latency_s=None,
                total_input_tokens=None,
                total_output_tokens=None,
            )

        passed = int(subset["passed"].sum())
        errors = int(subset["has_error"].sum())
        fails = total - passed
        acc = float(subset["score"].mean())
        fail_rate = float(errors / total)

        valid_lat = subset["latency_seconds"].dropna()
        # only consider latencies > 0 for stats
        valid_lat = valid_lat[valid_lat > 0]
        avg_lat = round(float(valid_lat.mean()), 4) if not valid_lat.empty else None
        p95_lat = round(float(valid_lat.quantile(0.95)), 4) if not valid_lat.empty else None

        valid_in = subset["input_tokens"].dropna()
        valid_out = subset["output_tokens"].dropna()
        tot_in = int(valid_in.sum()) if not valid_in.empty else None
        tot_out = int(valid_out.sum()) if not valid_out.empty else None

        return AggregateMetric(
            total_evaluations=total,
            accuracy=round(acc, 4),
            pass_count=passed,
            fail_count=fails,
            error_count=errors,
            failure_rate=round(fail_rate, 4),
            avg_latency_s=avg_lat,
            p95_latency_s=p95_lat,
            total_input_tokens=tot_in,
            total_output_tokens=tot_out,
        )

    # 1. Overall
    overall_metric = _to_metric(df)

    # 2. By model
    by_model: Dict[str, AggregateMetric] = {}
    for model_name, group in df.groupby("model"):
        by_model[str(model_name)] = _to_metric(group)

    # 3. By category
    by_category: Dict[str, AggregateMetric] = {}
    for cat_name, group in df.groupby("category"):
        by_category[str(cat_name)] = _to_metric(group)

    # 4. By model and category
    by_model_and_category: Dict[str, Dict[str, AggregateMetric]] = {}
    for model_name in by_model.keys():
        by_model_and_category[model_name] = {}
    for (model_name, cat_name), group in df.groupby(["model", "category"]):
        by_model_and_category[str(model_name)][str(cat_name)] = _to_metric(group)

    return StructuredMetrics(
        overall=overall_metric,
        by_model=by_model,
        by_category=by_category,
        by_model_and_category=by_model_and_category,
    )


def compare_models(
    run_result_or_evals: Union[RunResult, List[EvalResult]],
    by_category: bool = False,
    test_cases: Optional[List[TestCase]] = None,
) -> pd.DataFrame:
    """Computes comparative metrics across models using pandas.

    Metrics calculated per model (and per category if requested):
    - total_tests: count of evaluations
    - accuracy: mean score
    - pass_rate: percentage of tests where passed is True
    - failure_rate: percentage of tests with provider/eval errors
    - avg_latency_s: mean latency in seconds
    - p95_latency_s: 95th percentile latency in seconds
    - total_input_tokens: sum of input tokens (if available)
    - total_output_tokens: sum of output tokens (if available)

    Args:
        run_result_or_evals: RunResult instance or list of EvalResult.
        by_category: Whether to group metrics additionally by category.
        test_cases: Optional list of test cases to supply categories.

    Returns:
        pd.DataFrame with formatted summary metrics.
    """
    df = results_to_dataframe(run_result_or_evals, test_cases=test_cases)
    if df.empty:
        return pd.DataFrame()

    group_cols = ["model", "category"] if by_category else ["model"]

    def _agg_group(g: pd.DataFrame) -> pd.Series:
        total = len(g)
        errors = g["has_error"].sum()
        mean_score = g["score"].mean() if total > 0 else 0.0
        pass_rate = (g["passed"].sum() / total) if total > 0 else 0.0
        fail_rate = (errors / total) if total > 0 else 0.0

        latencies = g["latency_seconds"].dropna()
        avg_lat = latencies.mean() if not latencies.empty else 0.0
        p95_lat = latencies.quantile(0.95) if not latencies.empty else 0.0

        in_tok = g["input_tokens"].dropna().sum() if not g["input_tokens"].dropna().empty else None
        out_tok = g["output_tokens"].dropna().sum() if not g["output_tokens"].dropna().empty else None

        return pd.Series({
            "total_tests": total,
            "accuracy": round(float(mean_score), 4),
            "pass_rate": round(float(pass_rate), 4),
            "failure_rate": round(float(fail_rate), 4),
            "avg_latency_s": round(float(avg_lat), 4),
            "p95_latency_s": round(float(p95_lat), 4),
            "total_input_tokens": int(in_tok) if in_tok is not None else None,
            "total_output_tokens": int(out_tok) if out_tok is not None else None,
        })

    summary_df = df.groupby(group_cols, as_index=False, group_keys=False).apply(_agg_group, include_groups=False)
    return summary_df


def format_comparison_table(
    summary_df: pd.DataFrame,
    table_format: str = "simple",
) -> str:
    """Format the comparison summary DataFrame into a readable plain-text table."""
    if summary_df.empty:
        return "No evaluation results available."

    try:
        from tabulate import tabulate
        return tabulate(summary_df, headers="keys", tablefmt=table_format, showindex=False)
    except ImportError:
        return summary_df.to_string(index=False)
