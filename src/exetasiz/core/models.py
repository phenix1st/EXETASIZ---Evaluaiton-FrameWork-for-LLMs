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
