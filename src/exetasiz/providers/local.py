"""Adapter for OpenAI-compatible local LLM endpoints (e.g. Ollama, LM Studio, vLLM, LocalAI)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional
from exetasiz.providers.base import LLMProvider, ProviderResponse


class LocalModelProvider(LLMProvider):
    """Adapter for local endpoints serving OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/v1",
        name: Optional[str] = None,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        api_key: Optional[str] = "not-needed",
    ) -> None:
        super().__init__(name=name or f"local:{model}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key or "not-needed"

    def _generate_internal(self, prompt: str) -> ProviderResponse:
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        req_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            url=endpoint,
            data=req_data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                status_code = response.getcode()
                if status_code != 200:
                    raise RuntimeError(f"Local server returned HTTP {status_code}")
                res_body = response.read().decode("utf-8")
                res_data = json.loads(res_body)
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to local model endpoint at '{endpoint}'. "
                f"Ensure Ollama/LM Studio is running. Error: {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from local endpoint: {e}") from e

        choices = res_data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Local model response contained no choices: {res_data}")

        first_choice = choices[0]
        message = first_choice.get("message", {})
        text = message.get("content", "")

        usage = res_data.get("usage", {})
        in_tokens = usage.get("prompt_tokens")
        out_tokens = usage.get("completion_tokens")

        return ProviderResponse(
            text=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
