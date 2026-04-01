from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from backend.core.postprocess import process_text, process_transcription, words_to_text


# ----------------------------------------------------------------------
# Mock helper for Albert API
# ----------------------------------------------------------------------
def _mock_albert_response(prompt: str) -> Dict[str, Any]:
    # Simple deterministic responses based on prompt content
    if "Résume" in prompt:
        return {"choices": [{"message": {"content": "Résumé synthétique du texte."}}]}
    if "Liste les décisions" in prompt:
        return {"choices": [{"message": {"content": "- Action 1\n- Action 2"}}]}
    if "Supprime les tics" in prompt:
        return {"choices": [{"message": {"content": "Texte nettoyé sans tics."}}]}
    # Fallback
    return {"choices": [{"message": {"content": ""}}]}


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_words_to_text() -> None:
    words = [{"word": "Bonjour"}, {"word": "le"}, {"word": "monde"}]
    assert words_to_text(words) == "Bonjour le monde"


@pytest.mark.asyncio
@patch("backend.core.postprocess.requests.post")
async def test_process_text(mock_post: MagicMock) -> None:
    # Configure mock to return appropriate JSON depending on payload
    def side_effect(url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _mock_albert_response(json["messages"][0]["content"])
        return mock_resp
    mock_post.side_effect = side_effect

    sample_text = "euh Bonjour, alors je vais parler de la réunion."
    result = await process_text(sample_text)

    # Verify structure
    assert isinstance(result, dict)
    assert "summary" in result
    assert "action_points" in result
    assert "cleaned_text" in result

    # Verify mock responses were used
    assert result["summary"] == "Résumé synthétique du texte."
    assert result["action_points"] == ["Action 1", "Action 2"]
    assert result["cleaned_text"] == "Texte nettoyé sans tics."


@pytest.mark.asyncio
@patch("backend.core.postprocess.requests.post")
async def test_process_transcription(mock_post: MagicMock) -> None:
    # Same mock as above
    def side_effect(url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: float) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _mock_albert_response(json["messages"][0]["content"])
        return mock_resp
    mock_post.side_effect = side_effect

    words = [{"word": "euh"}, {"word": "Bonjour"}, {"word": "le"}, {"word": "monde"}]
    result = await process_transcription(words)

    assert isinstance(result, dict)
    assert result["summary"] == "Résumé synthétique du texte."
    assert result["action_points"] == ["Action 1", "Action 2"]
    assert result["cleaned_text"] == "Texte nettoyé sans tics."
