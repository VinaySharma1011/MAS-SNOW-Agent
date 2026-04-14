
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Assume the FastAPI app is defined in app.py as 'app'
# and the /agent/message endpoint is implemented there.
# If the app is in a different module, adjust the import accordingly.
from app import app

@pytest.fixture
def client():
    """Fixture to provide a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def valid_user_message():
    """Fixture for a valid user message to create a ticket."""
    return {
        "user_message": "Please create a ticket. short description: Printer not working, category: Hardware, priority: High"
    }

@pytest.fixture
def agent_response_success():
    """Fixture for a successful agent response."""
    return {
        "success": True,
        "response": "Ticket INC123456 created successfully.",
    }

@pytest.fixture
def agent_response_missing_fields():
    """Fixture for a response when required fields are missing."""
    return {
        "success": False,
        "error": "Missing required fields: short description, category, priority",
        "error_type": "validation"
    }

@pytest.fixture
def agent_response_auth_failure():
    """Fixture for a response when OAuth2 authentication fails."""
    return {
        "success": False,
        "error": "OAuth2 authentication failed",
        "error_type": "auth"
    }

@pytest.fixture
def agent_response_servicenow_unavailable():
    """Fixture for a response when ServiceNow API is unavailable."""
    return {
        "success": False,
        "error": "ServiceNow API unavailable",
        "error_type": "service_unavailable"
    }

class TestAgentMessageFunctional:
    def test_functional_create_ticket_via_api_endpoint(
        self,
        client,
        valid_user_message,
        agent_response_success
    ):
        """
        Functional test: Validates that a user can create a ServiceNow ticket by sending a properly formatted message
        to the /agent/message endpoint. Verifies HTTP 200, success=True, confirmation message, and no error fields.
        """
        # Patch the function that calls ServiceNow API to simulate ticket creation
        with patch("app.create_servicenow_ticket") as mock_create_ticket:
            mock_create_ticket.return_value = {
                "ticket_id": "INC123456",
                "status": "created"
            }
            # Patch authentication to always succeed
            with patch("app.get_oauth2_token") as mock_auth:
                mock_auth.return_value = "mocked_token"
                response = client.post("/agent/message", json=valid_user_message)
        
        assert response.status_code == 200, "Expected HTTP 200"
        data = response.json()
        assert data.get("success") is True, "Expected success=True"
        assert "Ticket" in data.get("response", ""), "Response should mention 'Ticket'"
        assert "created successfully" in data.get("response", ""), "Response should confirm creation"
        assert "error" not in data, "No error field should be present"
        assert "error_type" not in data, "No error_type field should be present"

    def test_functional_create_ticket_servicenow_api_unavailable(
        self,
        client,
        valid_user_message,
        agent_response_servicenow_unavailable
    ):
        """
        Functional test: Simulates ServiceNow API being unavailable when creating a ticket.
        Verifies error response and error_type.
        """
        with patch("app.create_servicenow_ticket") as mock_create_ticket:
            mock_create_ticket.side_effect = Exception("ServiceNow API unavailable")
            with patch("app.get_oauth2_token") as mock_auth:
                mock_auth.return_value = "mocked_token"
                response = client.post("/agent/message", json=valid_user_message)
        
        assert response.status_code == 200, "Expected HTTP 200 even on error"
        data = response.json()
        assert data.get("success") is False, "Expected success=False"
        assert "ServiceNow API unavailable" in data.get("error", ""), "Should report ServiceNow API unavailable"
        assert data.get("error_type") == "service_unavailable", "Error type should be 'service_unavailable'"

    def test_functional_create_ticket_missing_required_fields(
        self,
        client
    ):
        """
        Functional test: Simulates missing required fields in user_message.
        Verifies error response and error_type.
        """
        incomplete_message = {
            "user_message": "Please create a ticket."
        }
        with patch("app.create_servicenow_ticket") as mock_create_ticket:
            # Should not be called if validation fails
            mock_create_ticket.return_value = None
            with patch("app.get_oauth2_token") as mock_auth:
                mock_auth.return_value = "mocked_token"
                response = client.post("/agent/message", json=incomplete_message)
        
        assert response.status_code == 200, "Expected HTTP 200 even on validation error"
        data = response.json()
        assert data.get("success") is False, "Expected success=False"
        assert "Missing required fields" in data.get("error", ""), "Should report missing fields"
        assert data.get("error_type") == "validation", "Error type should be 'validation'"

    def test_functional_create_ticket_oauth2_authentication_failure(
        self,
        client,
        valid_user_message
    ):
        """
        Functional test: Simulates OAuth2 authentication failure when creating a ticket.
        Verifies error response and error_type.
        """
        with patch("app.get_oauth2_token") as mock_auth:
            mock_auth.side_effect = Exception("OAuth2 authentication failed")
            response = client.post("/agent/message", json=valid_user_message)
        
        assert response.status_code == 200, "Expected HTTP 200 even on auth error"
        data = response.json()
        assert data.get("success") is False, "Expected success=False"
        assert "OAuth2 authentication failed" in data.get("error", ""), "Should report auth failure"
        assert data.get("error_type") == "auth", "Error type should be 'auth'"

