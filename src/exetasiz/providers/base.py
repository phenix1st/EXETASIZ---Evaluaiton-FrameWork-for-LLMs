"""Base classes and response models for LLM providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ProviderResponse(BaseModel):
    """Response payload returned by an LLM provider."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="The generated text response")
    latency_seconds: float = Field(0.0, ge=0.0, description="Latency in seconds")
    input_tokens: Optional[int] = Field(default=None, description="Input token count if known")
    output_tokens: Optional[int] = Field(default=None, description="Output token count if known")


class LLMProvider(ABC):
    """Abstract base class for all LLM provider adapters."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, prompt: str) -> ProviderResponse:
        """Generates a response from the provider, measuring latency automatically.

        Args:
            prompt: Text prompt to provide to the model.

        Returns:
            ProviderResponse with latency_seconds measured.
        """
        start_time = time.perf_counter()
        response = self._generate_internal(prompt)
        duration = time.perf_counter() - start_time

        # If provider did not measure its own latency, set measured duration
        if response.latency_seconds == 0.0:
            response.latency_seconds = duration

        return response

    @abstractmethod
    def _generate_internal(self, prompt: str) -> ProviderResponse:
        """Internal generation method implemented by provider subclasses."""
        ...
