"""Provider registry and factory mechanism."""

from __future__ import annotations

from typing import Any, Callable, Dict, List
from exetasiz.providers.base import LLMProvider
from exetasiz.providers.mock import MockProvider
from exetasiz.providers.openai import OpenAIProvider
from exetasiz.providers.gemini import GeminiProvider
from exetasiz.providers.local import LocalModelProvider

_REGISTRY: Dict[str, Callable[..., LLMProvider]] = {}


def register_provider(name: str, factory_or_cls: Callable[..., LLMProvider]) -> None:
    """Register a provider constructor under a unique name."""
    _REGISTRY[name.lower()] = factory_or_cls


def get_provider(provider_type: str, **kwargs: Any) -> LLMProvider:
    """Instantiate a provider by name.

    Args:
        provider_type: Name of provider (e.g. 'mock', 'openai', 'gemini', 'local').
        **kwargs: Arguments passed to provider initialization.

    Returns:
        LLMProvider instance.

    Raises:
        ValueError: If provider is unknown.
    """
    key = provider_type.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider '{provider_type}'. Available providers: {available}"
        )
    return _REGISTRY[key](**kwargs)


def list_registered_providers() -> List[str]:
    """Return a sorted list of registered provider names."""
    return sorted(_REGISTRY.keys())


# Default registrations
register_provider("mock", MockProvider)
register_provider("openai", OpenAIProvider)
register_provider("gemini", GeminiProvider)
register_provider("local", LocalModelProvider)
