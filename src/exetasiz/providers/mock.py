"""Deterministic Mock LLM provider for tests and offline demonstrations."""

from __future__ import annotations

import time
from typing import Dict, Optional
from exetasiz.providers.base import LLMProvider, ProviderResponse


class MockProvider(LLMProvider):
    """Mock LLM Provider that maps prompts deterministically to preset answers."""

    def __init__(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "Mock response",
        fixed_latency: float = 0.01,
        simulate_error: bool = False,
        error_message: str = "Mock provider simulated error",
        input_tokens: Optional[int] = 10,
        output_tokens: Optional[int] = 5,
        **kwargs,
    ) -> None:
        display_name = name or model or "mock"
        super().__init__(name=display_name)
        self.model = model
        self.responses: Dict[str, str] = responses or {}
        self.default_response: str = default_response
        self.fixed_latency: float = fixed_latency
        self.simulate_error: bool = simulate_error
        self.error_message: str = error_message
        self.input_tokens: Optional[int] = input_tokens
        self.output_tokens: Optional[int] = output_tokens

    def _generate_internal(self, prompt: str) -> ProviderResponse:
        if self.simulate_error:
            raise RuntimeError(self.error_message)

        if self.fixed_latency > 0:
            time.sleep(self.fixed_latency)

        text = self.responses.get(prompt, self.default_response)
        return ProviderResponse(
            text=text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
