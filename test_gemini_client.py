"""Tests for gemini_client.py."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(text: str) -> MagicMock:
    """Build a minimal mock that looks like a genai Client response."""
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


# ---------------------------------------------------------------------------
# call_gemini
# ---------------------------------------------------------------------------

class TestCallGemini:
    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from gemini_client import call_gemini
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            call_gemini("hello")

    def test_returns_model_text(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_models = MagicMock()
        mock_models.generate_content.return_value = _make_response("Hello, world!")
        mock_client = MagicMock()
        mock_client.models = mock_models

        with patch("google.genai.Client", return_value=mock_client) as mock_cls:
            from gemini_client import call_gemini
            result = call_gemini("Say hello")

        mock_cls.assert_called_once_with(api_key="test-key")
        mock_models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash", contents="Say hello"
        )
        assert result == "Hello, world!"

    def test_custom_model_name(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_models = MagicMock()
        mock_models.generate_content.return_value = _make_response("Hi!")
        mock_client = MagicMock()
        mock_client.models = mock_models

        with patch("google.genai.Client", return_value=mock_client) as mock_cls:
            from gemini_client import call_gemini
            call_gemini("prompt", model_name="gemini-1.5-pro")

        mock_models.generate_content.assert_called_once_with(
            model="gemini-1.5-pro", contents="prompt"
        )


# ---------------------------------------------------------------------------
# main (CLI entry-point)
# ---------------------------------------------------------------------------

class TestMain:
    def test_prompt_from_argv(self, monkeypatch, capsys):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(sys, "argv", ["gemini_client.py", "What", "is", "AI?"])
        mock_models = MagicMock()
        mock_models.generate_content.return_value = _make_response("AI is...")
        mock_client = MagicMock()
        mock_client.models = mock_models

        with patch("google.genai.Client", return_value=mock_client):
            from gemini_client import main
            main()

        captured = capsys.readouterr()
        assert "AI is..." in captured.out

    def test_empty_prompt_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["gemini_client.py", "   "])
        from gemini_client import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_missing_api_key_exits(self, monkeypatch, capsys):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(sys, "argv", ["gemini_client.py", "hello"])
        from gemini_client import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "GEMINI_API_KEY" in captured.err
