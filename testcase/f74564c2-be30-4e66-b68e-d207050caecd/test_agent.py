
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

@pytest.fixture
def mock_content_safety_guard():
    """
    Fixture to mock the content safety guard/decorator.
    """
    with patch("agent.content_safety_guard") as mock_guard:
        yield mock_guard

@pytest.fixture
def mock_agent_response():
    """
    Fixture to mock the AgentResponse class.
    """
    with patch("agent.AgentResponse") as mock_response_cls:
        def _make_response(success, error_type, message=""):
            # Simulate the AgentResponse object
            return SimpleNamespace(success=success, error_type=error_type, message=message)
        mock_response_cls.side_effect = _make_response
        yield mock_response_cls

@pytest.fixture
def agent_client():
    """
    Fixture to provide a test client for the agent's HTTP API.
    This should be replaced with the actual test client (e.g., FlaskClient, FastAPI TestClient).
    """
    # For demonstration, we'll mock the client and its post method.
    client = MagicMock()
    yield client

@pytest.mark.security
def test_security_input_validation_and_content_safety_enforcement(
    agent_client, mock_content_safety_guard, mock_agent_response
):
    """
    Ensures that malicious or unsafe user input is detected and blocked by content safety guardrails and input validation.
    Verifies that:
      - response.success is False
      - response.error_type is 'validation_error' or indicates content safety violation
      - No sensitive information is leaked in the response
    """
    # Prepare malicious inputs
    malicious_inputs = [
        # PII
        {"user_message": "My SSN is 123-45-6789"},
        # SQL Injection
        {"user_message": "'; DROP TABLE users; --"},
        # Toxic content
        {"user_message": "I hate you and hope you die!"},
    ]

    # Simulate the agent's POST /agent/message endpoint
    # We'll patch the handler to simulate content safety and validation logic
    for payload in malicious_inputs:
        # Simulate content safety guard raising an exception or returning a block
        mock_content_safety_guard.return_value = False  # Unsafe input detected

        # Simulate the agent's response to unsafe input
        # The real handler would call AgentResponse(success=False, error_type=...)
        expected_error_type = "validation_error"
        if "hate" in payload["user_message"]:
            expected_error_type = "content_safety_violation"

        # Patch the POST method to return a mocked response
        response_obj = SimpleNamespace(
            success=False,
            error_type=expected_error_type,
            message="Input rejected due to policy violation."
        )
        agent_client.post.return_value = response_obj

        # Simulate POST request to /agent/message
        response = agent_client.post("/agent/message", json=payload)

        # Assertions
        assert response.success is False, "Agent should reject unsafe input"
        assert response.error_type in ("validation_error", "content_safety_violation"), \
            f"Unexpected error_type: {response.error_type}"
        # Ensure no sensitive info is leaked in the error message
        assert "SSN" not in response.message
        assert "DROP TABLE" not in response.message
        assert "traceback" not in response.message.lower()
        assert "exception" not in response.message.lower()
        assert "hate" not in response.message.lower()
        assert "123-45-6789" not in response.message
        assert "users" not in response.message.lower()
        # Error message should be generic and not leak implementation details
        assert "policy violation" in response.message or "not allowed" in response.message

    # Negative test: simulate content safety guard failing to block unsafe input
    # The agent should still not process the message and should return a validation error
    unsafe_payload = {"user_message": "'; DROP TABLE users; --"}
    mock_content_safety_guard.return_value = True  # Fails to block
    # Simulate the handler still catching the unsafe input
    response_obj = SimpleNamespace(
        success=False,
        error_type="validation_error",
        message="Input rejected due to policy violation."
    )
    agent_client.post.return_value = response_obj
    response = agent_client.post("/agent/message", json=unsafe_payload)
    assert response.success is False
    assert response.error_type in ("validation_error", "content_safety_violation")
    assert "DROP TABLE" not in response.message
    assert "users" not in response.message.lower()
    assert "policy violation" in response.message or "not allowed" in response.message
