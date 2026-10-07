"""OpenAI API provider adapter with lazy imports and clear error messaging."""

from __future__ import annotations

import os
from typing import Optional
from exetasiz.providers.base import LLMProvider, ProviderResponse


class OpenAIProvider(LLMProvider):
    """Adapter for OpenAI models (e.g. gpt-4o, gpt-4o-mini).

    Requires the 'openai' package and 'OPENAI_API_KEY' set in environment.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(name=name or f"openai:{model}")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI Python SDK is not installed. Please run: pip install openai"
            ) from e

        if not self.api_key:
            raise ValueError(
                "OpenAI API key missing. Please set the OPENAI_API_KEY environment variable "
                "or pass 'api_key' in the configuration."
            )

        self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def _generate_internal(self, prompt: str) -> ProviderResponse:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        in_tokens = response.usage.prompt_tokens if response.usage else None
        out_tokens = response.usage.completion_tokens if response.usage else None

        return ProviderResponse(
            text=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
