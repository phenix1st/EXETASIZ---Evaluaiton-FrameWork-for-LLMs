"""Tests for provider interface, registry, and mock provider."""

import pytest
from exetasiz.providers.base import LLMProvider, ProviderResponse
from exetasiz.providers.mock import MockProvider
from exetasiz.providers.registry import get_provider, list_registered_providers


def test_mock_provider_deterministic():
    provider = MockProvider(
        name="test-mock",
        responses={"ping": "pong"},
        default_response="unknown",
        fixed_latency=0.005,
    )
    resp = provider.generate("ping")
    assert resp.text == "pong"
    assert resp.latency_seconds > 0.0
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5

    resp_default = provider.generate("hello")
    assert resp_default.text == "unknown"


def test_mock_provider_simulated_error():
    provider = MockProvider(simulate_error=True, error_message="API Rate Limit")
    with pytest.raises(RuntimeError, match="API Rate Limit"):
        provider.generate("any prompt")


def test_provider_registry():
    prov = get_provider(
        "mock",
        name="custom_mock",
        responses={"1": "one"},
    )
    assert isinstance(prov, LLMProvider)
    assert prov.name == "custom_mock"
    assert prov.generate("1").text == "one"


def test_provider_registry_unknown():
    with pytest.raises(ValueError, match="Unknown provider 'unsupported'"):
        get_provider("unsupported")


def test_list_providers():
    providers = list_registered_providers()
    assert "mock" in providers
    assert "openai" in providers
    assert "gemini" in providers
    assert "local" in providers
