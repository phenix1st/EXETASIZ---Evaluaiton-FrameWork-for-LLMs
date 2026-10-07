"""Core domain models for test cases, datasets, and evaluation results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TestCase(BaseModel):
    """Represents an individual evaluation test case."""

    # Prevent pytest from attempting to collect TestCase as a test class
    __test__ = False

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique identifier for the test case")
    category: str = Field(..., description="Category or domain (e.g., reasoning, factual)")
    prompt: str = Field(..., description="The prompt input to be given to the LLM")
    expected_answer: str = Field(..., description="The reference/ground truth answer")
    context: Optional[str] = Field(default=None, description="Optional retrieval or background context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata")


class Dataset(BaseModel):
    """Represents a collection of test cases under an identified dataset name."""

    name: str = Field(..., description="Dataset name")
    test_cases: List[TestCase] = Field(..., description="List of test cases")


class EvalResult(BaseModel):
    """Result of running an evaluator on a generated answer for a single test case."""

    test_case_id: str = Field(..., description="ID of the test case")
    model: str = Field(..., description="Model identifier")
    generated_answer: str = Field(..., description="Generated answer from the model")
    score: float = Field(..., ge=0.0, le=1.0, description="Evaluation score from 0.0 to 1.0")
    passed: bool = Field(..., description="Whether the test passed based on evaluator threshold")
    evaluator: str = Field(..., description="Name of the evaluator used")
    latency_seconds: float = Field(..., ge=0.0, description="Provider call latency in seconds")
    input_tokens: Optional[int] = Field(default=None, description="Input token count if available")
    output_tokens: Optional[int] = Field(default=None, description="Output token count if available")
    error: Optional[str] = Field(default=None, description="Error message if the provider or evaluation failed")


class RunResult(BaseModel):
    """Consolidated results of an entire evaluation experiment run."""

    experiment_name: str = Field(..., description="Name of the experiment")
    timestamp: str = Field(..., description="ISO 8601 timestamp when the run completed")
    results: List[EvalResult] = Field(..., description="List of individual evaluation results")
    config_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Snapshot of the run configuration")


class DatasetMetadata(BaseModel):
    """Metadata describing the dataset used in an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Dataset name")
    path: str = Field(..., description="File path to the dataset")
    num_test_cases: int = Field(..., ge=0, description="Total number of test cases")


class ResultsEnvelope(BaseModel):
    """Envelope wrapping results.json for structured persistence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0", description="Schema version of results envelope")
    run_id: str = Field(..., description="Unique run identifier")
    experiment_name: str = Field(..., description="Experiment name")
    started_at: str = Field(..., description="ISO 8601 timestamp when the run started")
    finished_at: str = Field(..., description="ISO 8601 timestamp when the run completed")
    duration_seconds: float = Field(..., ge=0.0, description="Total duration of run in seconds")
    dataset: DatasetMetadata = Field(..., description="Dataset metadata")
    models: List[str] = Field(..., description="List of evaluated model names")
    evaluators: List[str] = Field(..., description="List of evaluator names used")
    exetasiz_version: str = Field(..., description="Version of Exetasiz that executed the run")
    results: List[EvalResult] = Field(..., description="Individual evaluation results")


class AggregateMetric(BaseModel):
    """Summary metrics for a group of evaluation results."""

    model_config = ConfigDict(extra="forbid")

    total_evaluations: int = Field(..., ge=0, description="Total evaluations in this group")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Mean score (accuracy)")
    pass_count: int = Field(..., ge=0, description="Number of evaluations marked passed")
    fail_count: int = Field(..., ge=0, description="Number of failed evaluations")
    error_count: int = Field(..., ge=0, description="Number of provider/evaluator errors")
    failure_rate: float = Field(..., ge=0.0, le=1.0, description="Error failure rate")
    avg_latency_s: Optional[float] = Field(default=None, description="Average latency in seconds")
    p95_latency_s: Optional[float] = Field(default=None, description="95th percentile latency in seconds")
    total_input_tokens: Optional[int] = Field(default=None, description="Sum of input tokens")
    total_output_tokens: Optional[int] = Field(default=None, description="Sum of output tokens")


class StructuredMetrics(BaseModel):
    """Hierarchical metrics document saved in metrics.json."""

    model_config = ConfigDict(extra="forbid")

    overall: AggregateMetric = Field(..., description="Overall metrics across all models and test cases")
    by_model: Dict[str, AggregateMetric] = Field(..., description="Metrics aggregated per model")
    by_category: Dict[str, AggregateMetric] = Field(..., description="Metrics aggregated per category")
    by_model_and_category: Dict[str, Dict[str, AggregateMetric]] = Field(
        ..., description="Metrics grouped by model and category"
    )


class LoadedRun(BaseModel):
    """Container for an in-memory loaded run folder."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    envelope: ResultsEnvelope = Field(..., description="Parsed results envelope")
    metrics: StructuredMetrics = Field(..., description="Parsed structured metrics")
    config_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Parsed config snapshot")
    run_dir: str = Field(..., description="Absolute string path to run directory")
