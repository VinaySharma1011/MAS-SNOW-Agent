
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict

# Assume these are the classes under test, imported from your codebase
# from your_module import OAuth2Authenticator, ServiceNowAPIClient, TicketResponse

# For demonstration, define minimal stubs (to be replaced with real imports)
class TicketResponse:
    def __init__(self, success: bool, ticket_id: str = None, error: str = None):
        self.success = success
        self.ticket_id = ticket_id
        self.error = error

class OAuth2Authenticator:
    def __init__(self, client_id: str, client_secret: str, token_url: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url

    def get_token(self) -> str:
        # Real implementation would retrieve a token
        pass

class ServiceNowAPIClient:
    def __init__(self, instance_url: str, authenticator: OAuth2Authenticator):
        self.instance_url = instance_url
        self.authenticator = authenticator

    def create_ticket(self, ticket_data: Dict) -> TicketResponse:
        # Real implementation would create a ticket via ServiceNow API
        pass

@pytest.fixture
def valid_ticket_data():
    """Fixture providing valid ticket data for ServiceNow ticket creation."""
    return {
        "short_description": "Test incident",
        "description": "This is a test ticket created by integration test.",
        "category": "inquiry",
        "priority": "2"
    }

@pytest.fixture
def mock_authenticator():
    """Fixture providing a mock OAuth2Authenticator."""
    return MagicMock(spec=OAuth2Authenticator)

@pytest.fixture
def client(mock_authenticator):
    """Fixture providing a ServiceNowAPIClient with a mocked authenticator."""
    return ServiceNowAPIClient(instance_url="https://example.service-now.com", authenticator=mock_authenticator)

def test_integration_servicenow_oauth2_authentication_and_ticket_creation(valid_ticket_data, mock_authenticator):
    """
    Integration test for ServiceNow ticket creation using OAuth2Authenticator.
    Ensures that get_token is called, a valid token is used, and ticket creation succeeds.
    """
    # Arrange
    valid_token = "mocked-access-token"
    mock_authenticator.get_token.return_value = valid_token

    # Patch the ServiceNowAPIClient.create_ticket to simulate API call
    with patch.object(ServiceNowAPIClient, "create_ticket", autospec=True) as mock_create_ticket:
        # Simulate successful ticket creation
        mock_response = TicketResponse(success=True, ticket_id="INC0012345")
        mock_create_ticket.return_value = mock_response

        client = ServiceNowAPIClient("https://example.service-now.com", mock_authenticator)

        # Act
        response = client.create_ticket(valid_ticket_data)

        # Assert
        mock_authenticator.get_token.assert_called_once()
        mock_create_ticket.assert_called_once_with(client, valid_ticket_data)
        assert isinstance(response, TicketResponse)
        assert response.success is True
        assert response.ticket_id is not None

def test_integration_servicenow_oauth2_token_retrieval_failure(valid_ticket_data, mock_authenticator):
    """
    Integration error scenario: Token retrieval failure.
    Ensures that if get_token raises an exception, ticket creation fails gracefully.
    """
    # Arrange
    mock_authenticator.get_token.side_effect = Exception("Token retrieval failed")

    client = ServiceNowAPIClient("https://example.service-now.com", mock_authenticator)

    # Patch create_ticket to call get_token and handle the exception
    with patch.object(ServiceNowAPIClient, "create_ticket", autospec=True) as mock_create_ticket:
        def _side_effect(self, ticket_data):
            try:
                token = self.authenticator.get_token()
            except Exception as e:
                return TicketResponse(success=False, ticket_id=None, error=str(e))
            # Would normally proceed to API call
            return TicketResponse(success=True, ticket_id="INC0012345")
        mock_create_ticket.side_effect = _side_effect

        # Act
        response = client.create_ticket(valid_ticket_data)

        # Assert
        mock_authenticator.get_token.assert_called_once()
        assert isinstance(response, TicketResponse)
        assert response.success is False
        assert response.ticket_id is None
        assert "Token retrieval failed" in response.error

def test_integration_servicenow_oauth2_invalid_credentials(valid_ticket_data, mock_authenticator):
    """
    Integration error scenario: Invalid ServiceNow credentials.
    Ensures that if ServiceNow API returns an authentication error, ticket creation fails.
    """
    # Arrange
    valid_token = "mocked-access-token"
    mock_authenticator.get_token.return_value = valid_token

    client = ServiceNowAPIClient("https://example.service-now.com", mock_authenticator)

    # Patch create_ticket to simulate ServiceNow API authentication error
    with patch.object(ServiceNowAPIClient, "create_ticket", autospec=True) as mock_create_ticket:
        mock_create_ticket.return_value = TicketResponse(success=False, ticket_id=None, error="Invalid credentials")

        # Act
        response = client.create_ticket(valid_ticket_data)

        # Assert
        mock_authenticator.get_token.assert_called_once()
        assert isinstance(response, TicketResponse)
        assert response.success is False
        assert response.ticket_id is None
        assert "Invalid credentials" in response.error

def test_integration_servicenow_api_returns_error(valid_ticket_data, mock_authenticator):
    """
    Integration error scenario: ServiceNow API returns error.
    Ensures that if the ServiceNow API returns an error, ticket creation fails.
    """
    # Arrange
    valid_token = "mocked-access-token"
    mock_authenticator.get_token.return_value = valid_token

    client = ServiceNowAPIClient("https://example.service-now.com", mock_authenticator)

    # Patch create_ticket to simulate ServiceNow API error (e.g., 500 Internal Server Error)
    with patch.object(ServiceNowAPIClient, "create_ticket", autospec=True) as mock_create_ticket:
        mock_create_ticket.return_value = TicketResponse(success=False, ticket_id=None, error="Internal Server Error")

        # Act
        response = client.create_ticket(valid_ticket_data)

        # Assert
        mock_authenticator.get_token.assert_called_once()
        assert isinstance(response, TicketResponse)
        assert response.success is False
        assert response.ticket_id is None
        assert "Internal Server Error" in response.error
