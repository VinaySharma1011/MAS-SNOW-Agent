
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

@pytest.fixture
def mock_agent_response():
    """
    Fixture to provide a mock AgentResponse class for testing.
    """
    class AgentResponse:
        def __init__(self, success, error_type, error_message):
            self.success = success
            self.error_type = error_type
            self.error_message = error_message
    return AgentResponse

@pytest.fixture
def agent_api_client():
    """
    Fixture to provide a mock API client for the agent.
    """
    class MockClient:
        def post(self, endpoint, json=None):
            # Simulate validation error for malformed JSON or missing fields
            if json is None or 'message' not in json:
                # Simulate a safe error response (no stack trace, no sensitive info)
                return SimpleNamespace(
                    status_code=400,
                    json=lambda: {
                        "success": False,
                        "error_type": "validation_error",
                        "error_message": "Invalid request payload"
                    }
                )
            # Simulate normal response (should not be reached in this test)
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "success": True,
                    "result": "ok"
                }
            )
    return MockClient()

@pytest.mark.security
def test_security_api_endpoint_unauthorized_access(agent_api_client, mock_agent_response):
    """
    Ensures that endpoints do not leak sensitive information or stack traces on unauthorized or malformed requests.
    """
    # Simulate malformed JSON (missing required fields)
    with patch("builtins.print"):  # Patch print to avoid debug output
        response_obj = agent_api_client.post("/agent/message", json={})
        response_json = response_obj.json()

        # Create AgentResponse instance for assertion
        response = mock_agent_response(
            success=response_json.get("success"),
            error_type=response_json.get("error_type"),
            error_message=response_json.get("error_message")
        )

        # Assert success is False
        assert response.success is False, "Expected success to be False for malformed request"

        # Assert error_type is 'validation_error' or 'http_error'
        assert response.error_type in ("validation_error", "http_error"), \
            f"Expected error_type to be 'validation_error' or 'http_error', got {response.error_type}"

        # Assert no stack trace or sensitive config is present in the error message
        error_message = response.error_message or ""
        sensitive_keywords = [
            "Traceback", "Exception", "KeyError", "os.environ", "SECRET", "PASSWORD", "token", "config"
        ]
        for keyword in sensitive_keywords:
            assert keyword not in error_message, f"Sensitive info '{keyword}' leaked in error message: {error_message}"

        # Simulate a scenario where stack trace is returned (should fail)
        bad_response_json = {
            "success": False,
            "error_type": "validation_error",
            "error_message": "Traceback (most recent call last):\n  File \"agent.py\", line 42, in handle\n    ...\nKeyError: 'message'"
        }
        bad_response = mock_agent_response(
            success=bad_response_json["success"],
            error_type=bad_response_json["error_type"],
            error_message=bad_response_json["error_message"]
        )
        for keyword in sensitive_keywords:
            assert keyword not in bad_response.error_message, (
                f"Test should fail if sensitive info '{keyword}' is present in error message"
            )

        # Simulate a scenario where sensitive env variable is leaked (should fail)
        leaked_env_response_json = {
            "success": False,
            "error_type": "validation_error",
            "error_message": "Missing field. Config: SECRET_KEY=supersecret"
        }
        leaked_env_response = mock_agent_response(
            success=leaked_env_response_json["success"],
            error_type=leaked_env_response_json["error_type"],
            error_message=leaked_env_response_json["error_message"]
        )
        for keyword in sensitive_keywords:
            assert keyword not in leaked_env_response.error_message, (
                f"Test should fail if sensitive info '{keyword}' is present in error message"
            )
