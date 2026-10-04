"""Tests for Google Gemini integration and fallback chain in LLMClient."""

from __future__ import annotations

import os
from unittest import mock
import pytest

from backend.services.llm_client import (
    GEMINI,
    GROQ,
    OLLAMA,
    OFFLINE,
    LLMClient,
    LLMResponse,
    _build_chain,
)


def test_chain_order_default_groq():
    with mock.patch.dict(os.environ, {"DEFAULT_LLM": "groq"}):
        chain = _build_chain()
        assert chain == [GROQ, GEMINI, OLLAMA]


def test_chain_order_forced_gemini():
    with mock.patch.dict(os.environ, {"DEFAULT_LLM": "gemini"}):
        chain = _build_chain()
        assert chain == [GEMINI, GROQ, OLLAMA]


def test_health_check_gemini_entry():
    client = LLMClient()
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test_gemini_key", "GEMINI_MODEL": "gemini-1.5-flash"}):
        health = client.health_check()
        assert GEMINI in health
        assert health[GEMINI]["configured"] is True
        assert health[GEMINI]["reachable"] is True
        assert health[GEMINI]["model"] == "gemini-1.5-flash"


def test_fallback_to_gemini_when_groq_empty():
    client = LLMClient()
    mock_gemini_resp = LLMResponse(
        text="Response from Gemini 1.5 Flash",
        provider=GEMINI,
        model="gemini-1.5-flash",
    )
    with mock.patch.dict(os.environ, {"GROQ_API_KEY": "", "GEMINI_API_KEY": "valid_gemini_key"}):
        with mock.patch("backend.services.llm_client.try_gemini", return_value=mock_gemini_resp) as mock_gemini:
            resp = client.generate("Explain RAG")
            assert resp.ok is True
            assert resp.provider == GEMINI
            assert resp.model == "gemini-1.5-flash"
            assert resp.text == "Response from Gemini 1.5 Flash"
            assert any("groq -> GROQ_API_KEY not configured" in att for att in resp.attempts)
            mock_gemini.assert_called_once()


def test_fallback_through_gemini_when_gemini_fails():
    client = LLMClient()
    with mock.patch.dict(os.environ, {"GROQ_API_KEY": "", "GEMINI_API_KEY": "bad_key"}):
        with mock.patch("backend.services.llm_client.try_gemini", side_effect=RuntimeError("Invalid API key")):
            with mock.patch("backend.services.llm_client.ollama_available", return_value=False):
                resp = client.generate("Explain RAG")
                assert resp.ok is False
                assert resp.provider == OFFLINE
                assert any("gemini -> RuntimeError" in att for att in resp.attempts)
