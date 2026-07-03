"""
tests/test_inference.py

Unit tests for OllamaClient — mocked HTTP, no Ollama required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.inference.ollama_client import OllamaClient


def _make_client(model: str = "test-model") -> OllamaClient:
    """Construct OllamaClient with _check_connection stubbed out."""
    with patch.object(OllamaClient, "_check_connection"):
        return OllamaClient(model=model)


class TestGenerate:
    def test_returns_stripped_response(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "  answer text  "}
        with patch("src.inference.ollama_client.requests.post", return_value=mock_resp):
            result = client.generate("What is Class B airspace?")
        assert result == "answer text"

    def test_passes_model_name(self) -> None:
        client = _make_client(model="aerosense-chartqa-qlora-fixed")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        with patch("src.inference.ollama_client.requests.post", return_value=mock_resp) as mock_post:
            client.generate("question")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "aerosense-chartqa-qlora-fixed"

    def test_stream_is_false(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        with patch("src.inference.ollama_client.requests.post", return_value=mock_resp) as mock_post:
            client.generate("question")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["stream"] is False

    def test_max_tokens_forwarded(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        with patch("src.inference.ollama_client.requests.post", return_value=mock_resp) as mock_post:
            client.generate("question", max_tokens=256)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"]["num_predict"] == 256

    def test_raises_on_http_error(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
        with patch("src.inference.ollama_client.requests.post", return_value=mock_resp):
            with pytest.raises(Exception, match="500"):
                client.generate("question")

    def test_prompt_includes_question(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        with patch("src.inference.ollama_client.requests.post", return_value=mock_resp) as mock_post:
            client.generate("What is RNAV?")
        payload = mock_post.call_args.kwargs["json"]
        assert "What is RNAV?" in payload["prompt"]

    def test_context_included_in_prompt(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        with patch("src.inference.ollama_client.requests.post", return_value=mock_resp) as mock_post:
            client.generate("question", context="FAA AIM 3-2-1")
        payload = mock_post.call_args.kwargs["json"]
        assert "FAA AIM 3-2-1" in payload["prompt"]


class TestCheckConnection:
    def test_warns_when_model_not_found(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "other-model:latest"}]}
        with patch("src.inference.ollama_client.requests.get", return_value=mock_resp):
            with patch("src.inference.ollama_client.logger") as mock_logger:
                OllamaClient(model="missing-model")
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "missing-model" in warning_msg

    def test_logs_success_when_model_present(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "aerosense-chartqa:latest"}]}
        with patch("src.inference.ollama_client.requests.get", return_value=mock_resp):
            with patch("src.inference.ollama_client.logger") as mock_logger:
                OllamaClient(model="aerosense-chartqa")
        mock_logger.info.assert_called_once()

    def test_warns_when_ollama_unreachable(self) -> None:
        with patch(
            "src.inference.ollama_client.requests.get",
            side_effect=Exception("Connection refused"),
        ):
            with patch("src.inference.ollama_client.logger") as mock_logger:
                OllamaClient(model="any-model")
        mock_logger.warning.assert_called_once()

    def test_matches_model_without_latest_tag(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "aerosense-chartqa:latest"}]}
        with patch("src.inference.ollama_client.requests.get", return_value=mock_resp):
            with patch("src.inference.ollama_client.logger") as mock_logger:
                OllamaClient(model="aerosense-chartqa")
        mock_logger.info.assert_called_once()
