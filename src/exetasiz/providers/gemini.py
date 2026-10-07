"""Google Gemini API provider adapter with lazy imports and clear error messaging."""

from __future__ import annotations

import os
from typing import Optional
from exetasiz.providers.base import LLMProvider, ProviderResponse


class GeminiProvider(LLMProvider):
    """Adapter for Google Gemini models (e.g. gemini-1.5-flash, gemini-2.0-flash).

    Requires 'google-genai' or 'google-generativeai' and 'GEMINI_API_KEY' in environment.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(name=name or f"gemini:{model}")
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self._model_instance = None

    def _get_model(self):
        if self._model_instance is not None:
            return self._model_instance

        if not self.api_key:
            raise ValueError(
                "Gemini API key missing. Please set the GEMINI_API_KEY environment variable "
                "or pass 'api_key' in the configuration."
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model_instance = genai.GenerativeModel(
                model_name=self.model,
                generation_config={"temperature": self.temperature},
            )
            return self._model_instance
        except ImportError as e:
            raise ImportError(
                "Google Generative AI SDK is not installed. Please run: pip install google-generativeai"
            ) from e

    def _generate_internal(self, prompt: str) -> ProviderResponse:
        model = self._get_model()
        response = model.generate_content(prompt)
        text = response.text or ""

        # Extract token usage if available in response metadata
        in_tokens = None
        out_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            in_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            out_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        return ProviderResponse(
            text=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
