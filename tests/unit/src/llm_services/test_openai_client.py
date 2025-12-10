import pytest
import json
from unittest.mock import patch, MagicMock
import httpx

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

# Import the client from your source, and the exceptions from the library
from src.llm_services.openai_client import OpenAIClient
from openai import APIError, APITimeoutError

# --- Fixtures for Testing OpenAIClient ---

@pytest.fixture
def base_config():
    """Provides a base configuration for the client."""
    return {
        "model": "test-model",
        "temperature": 0.5,
        "request_settings": {
            "max_retries": 2,
            "initial_timeout_seconds": 5,
            "backoff_factor": 1.5
        }
    }

@pytest.fixture
def mock_success_response():
    """Creates a mock successful API response object."""
    response = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response_content = json.dumps({"entities": [{"text": "finding", "label": "FIND"}]})
    response.choices = [MagicMock(message=MagicMock(content=response_content))]
    return response

# --- Initialization Tests ---

@patch('src.llm_services.openai_client.OpenAI')
def test_initialization_with_api_key(mock_openai_constructor, base_config):
    """
    Tests that the client initializes correctly when an API key is provided directly.
    """
    client = OpenAIClient(config=base_config, api_key="test_key")
    mock_openai_constructor.assert_called_once_with(api_key="test_key")

@patch('src.llm_services.openai_client.OpenAI')
def test_initialization_with_env_var(mock_openai_constructor, base_config, monkeypatch):
    """
    Tests that the client initializes correctly using an environment variable for the API key.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "env_test_key")
    client = OpenAIClient(config=base_config)
    mock_openai_constructor.assert_called_once_with(api_key="env_test_key")

def test_initialization_raises_error_without_key(base_config, monkeypatch):
    """
    Tests that a ValueError is raised if no API key is provided.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        OpenAIClient(config=base_config)

# --- Prediction and Error Handling Tests ---

def test_get_ner_prediction_success(base_config, mock_success_response, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    client = OpenAIClient(config=base_config)
    with patch.object(client.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = mock_success_response
        result = client.get_ner_prediction("some prompt")
    assert result == [{"text": "finding", "label": "FIND"}]
    mock_create.assert_called_once()

def test_get_ner_prediction_handles_api_error(base_config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    client = OpenAIClient(config=base_config)
    with patch.object(client.client.chat.completions, 'create') as mock_create:
        mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_create.side_effect = APIError("API Error", request=mock_request, body=None)
        result = client.get_ner_prediction("some prompt")
    assert result == []

def test_get_ner_prediction_handles_json_decode_error(base_config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    client = OpenAIClient(config=base_config)
    with patch.object(client.client.chat.completions, 'create') as mock_create:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="this is not json"))]
        mock_response.usage = None
        mock_create.return_value = mock_response
        result = client.get_ner_prediction("some prompt")
    assert result == []

def test_get_ner_prediction_retries_on_timeout_and_succeeds(base_config, mock_success_response, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    client = OpenAIClient(config=base_config)
    with patch.object(client.client.chat.completions, 'create') as mock_create:
        mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_create.side_effect = [
            APITimeoutError(request=mock_request),
            mock_success_response
        ]
        result = client.get_ner_prediction("some prompt")
    assert mock_create.call_count == 2
    assert result == [{"text": "finding", "label": "FIND"}]

def test_get_ner_prediction_fails_after_max_retries_on_timeout(base_config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    client = OpenAIClient(config=base_config)
    with patch.object(client.client.chat.completions, 'create') as mock_create:
        max_attempts = base_config["request_settings"]["max_retries"] + 1
        mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_create.side_effect = [APITimeoutError(request=mock_request)] * max_attempts
        result = client.get_ner_prediction("some prompt")
    assert mock_create.call_count == max_attempts
    assert result == []