
import pytest
from unittest.mock import patch, MagicMock

# Assume the following imports are from the agent implementation
# from agent import Agent, AgentResponse, ServiceNowAPIClient

@pytest.fixture
def agent_instance():
    """
    Fixture to provide an Agent instance with mocked dependencies.
    """
    # We'll patch ServiceNowAPIClient in the test, so just return a basic Agent
    # If Agent takes dependencies in constructor, patch as needed
    from agent import Agent
    return Agent()

@pytest.fixture
def mock_intent_classifier():
    """
    Fixture to mock the agent's intent classifier.
    """
    with patch("agent.Agent._classify_intent") as mock_classifier:
        yield mock_classifier

@pytest.fixture
def mock_ticket_validator():
    """
    Fixture to mock the agent's ticket ID validator.
    """
    with patch("agent.Agent._validate_ticket_id") as mock_validator:
        yield mock_validator

@pytest.fixture
def mock_servicenow_client():
    """
    Fixture to mock the ServiceNowAPIClient.get_ticket_status method.
    """
    with patch("agent.ServiceNowAPIClient.get_ticket_status") as mock_get_status:
        yield mock_get_status

@pytest.fixture
def agent_response_class():
    """
    Fixture to import AgentResponse class.
    """
    from agent import AgentResponse
    return AgentResponse

def test_integration_ticket_status_retrieval_workflow(
    agent_instance,
    mock_intent_classifier,
    mock_ticket_validator,
    mock_servicenow_client,
    agent_response_class
):
    """
    Integration test: Tests the end-to-end workflow for retrieving the status of an existing ticket,
    including intent classification, validation, ServiceNow API call, and response formatting.
    """
    user_message = "What is the status of ticket INC123456?"

    # Arrange: Set up mocks for intent, validation, and ServiceNow API
    mock_intent_classifier.return_value = "GET_STATUS"
    mock_ticket_validator.return_value = True
    mock_servicenow_client.return_value = {
        "ticket_id": "INC123456",
        "status": "In Progress",
        "short_description": "User cannot access email"
    }

    # Act: Simulate POST /agent/message
    # Assuming agent_instance.handle_message is the entry point
    response = agent_instance.handle_message(user_message)

    # Assert: Intent is classified as GET_STATUS
    mock_intent_classifier.assert_called_once_with(user_message)
    # Assert: Validation passes for ticket_id
    mock_ticket_validator.assert_called_once_with("INC123456")
    # Assert: ServiceNowAPIClient.get_ticket_status is called with correct ticket_id
    mock_servicenow_client.assert_called_once_with("INC123456")
    # Assert: AgentResponse with success=True and response contains the ticket status
    assert isinstance(response, agent_response_class)
    assert response.success is True
    assert "status of ticket INC123456 is" in response.response.lower()

def test_integration_ticket_status_retrieval_ticket_not_found(
    agent_instance,
    mock_intent_classifier,
    mock_ticket_validator,
    mock_servicenow_client,
    agent_response_class
):
    """
    Integration test: Error scenario where the ticket ID is not found in ServiceNow.
    """
    user_message = "What is the status of ticket INC999999?"

    mock_intent_classifier.return_value = "GET_STATUS"
    mock_ticket_validator.return_value = True
    mock_servicenow_client.return_value = None  # Simulate not found

    response = agent_instance.handle_message(user_message)

    mock_intent_classifier.assert_called_once_with(user_message)
    mock_ticket_validator.assert_called_once_with("INC999999")
    mock_servicenow_client.assert_called_once_with("INC999999")
    assert isinstance(response, agent_response_class)
    assert response.success is False
    assert "not found" in response.response.lower()

def test_integration_ticket_status_retrieval_servicenow_api_error(
    agent_instance,
    mock_intent_classifier,
    mock_ticket_validator,
    mock_servicenow_client,
    agent_response_class
):
    """
    Integration test: Error scenario where the ServiceNow API returns an error.
    """
    user_message = "What is the status of ticket INC123456?"

    mock_intent_classifier.return_value = "GET_STATUS"
    mock_ticket_validator.return_value = True
    mock_servicenow_client.side_effect = Exception("ServiceNow API error")

    response = agent_instance.handle_message(user_message)

    mock_intent_classifier.assert_called_once_with(user_message)
    mock_ticket_validator.assert_called_once_with("INC123456")
    mock_servicenow_client.assert_called_once_with("INC123456")
    assert isinstance(response, agent_response_class)
    assert response.success is False
    assert "error" in response.response.lower() or "unable" in response.response.lower()

def test_integration_ticket_status_retrieval_malformed_ticket_id(
    agent_instance,
    mock_intent_classifier,
    mock_ticket_validator,
    agent_response_class
):
    """
    Integration test: Error scenario where the ticket ID is malformed.
    ServiceNow API should NOT be called.
    """
    user_message = "What is the status of ticket 123ABC?"

    mock_intent_classifier.return_value = "GET_STATUS"
    mock_ticket_validator.return_value = False  # Malformed ticket ID

    with patch("agent.ServiceNowAPIClient.get_ticket_status") as mock_get_status:
        response = agent_instance.handle_message(user_message)
        mock_intent_classifier.assert_called_once_with(user_message)
        mock_ticket_validator.assert_called_once_with("123ABC")
        mock_get_status.assert_not_called()
        assert isinstance(response, agent_response_class)
        assert response.success is False
        assert "invalid" in response.response.lower() or "malformed" in response.response.lower()
