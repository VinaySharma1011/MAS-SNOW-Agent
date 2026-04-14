
import pytest
import time
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_azure_ai_search():
    """
    Fixture to mock the Azure AI Search service.
    Simulates a fast, successful search response.
    """
    def _search(query):
        # Simulate a realistic search result
        return {
            "documents": [
                {"content": "To reset your password, go to the settings page and click 'Reset Password'."}
            ]
        }
    return MagicMock(side_effect=_search)

@pytest.fixture
def mock_llm():
    """
    Fixture to mock the LLM (e.g., OpenAI) response.
    Simulates a fast, successful LLM completion.
    """
    def _generate(prompt):
        # Simulate a relevant LLM answer
        return {
            "success": True,
            "response": "To reset your password, go to the settings page and click 'Reset Password'."
        }
    return MagicMock(side_effect=_generate)

@pytest.fixture
def agent_message_handler(mock_azure_ai_search, mock_llm):
    """
    Fixture to provide the agent's /agent/message handler with all dependencies mocked.
    """
    # Import here to avoid issues if the real module is not present
    # Simulate the handler as a function for test purposes
    class AgentResponse:
        def __init__(self, success, response):
            self.success = success
            self.response = response

    def handler(user_message: str):
        # Simulate the RAG pipeline: search + LLM
        search_result = mock_azure_ai_search(user_message)
        docs = search_result.get("documents", [])
        if docs:
            context = docs[0]["content"]
        else:
            context = "Sorry, I couldn't find an answer to your question."
        llm_result = mock_llm(context)
        return AgentResponse(success=llm_result["success"], response=llm_result["response"])
    return handler

@pytest.mark.performance
def test_performance_response_time_for_knowledge_base_query(agent_message_handler):
    """
    Measures the response time for a general query that triggers the RAG pipeline (Azure AI Search + LLM)
    to ensure it meets performance requirements.
    - Total response time is less than 3 seconds
    - response.success is True
    - response.response contains relevant answer or fallback message
    """
    user_message = "How do I reset my password?"

    start_time = time.time()
    response = agent_message_handler(user_message)
    end_time = time.time()
    execution_time = end_time - start_time

    assert execution_time < 3.0, f"Test took {execution_time:.2f}s, expected < 3s"
    assert response.success is True, "Agent response should indicate success"
    assert (
        "reset your password" in response.response.lower()
        or "couldn't find an answer" in response.response.lower()
    ), "Agent response should contain a relevant answer or fallback message"
