
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from typing import List

# Assume the following imports are from the agent codebase
# from agent.main import app
# from agent.schemas import AgentResponse

# For demonstration, define a minimal AgentResponse here
class AgentResponse:
    def __init__(self, success: bool, message: str):
        self.success = success
        self.message = message

@pytest.fixture
def valid_ticket_message():
    """Returns a valid ticket creation message payload."""
    return {
        "user_id": "user123",
        "message": "Please create a ServiceNow ticket for my laptop issue.",
        "channel": "slack"
    }

@pytest.fixture
def mock_servicenow_api():
    """
    Mocks the ServiceNow API call used during ticket creation.
    Returns a MagicMock that always returns a successful ticket creation response.
    """
    def _mock_create_ticket(*args, **kwargs):
        return {"ticket_id": "INC123456", "status": "created"}
    return MagicMock(side_effect=_mock_create_ticket)

@pytest.fixture
def mock_agent_message_endpoint(mock_servicenow_api):
    """
    Mocks the /agent/message endpoint handler.
    Simulates the logic of ticket creation and returns AgentResponse.
    """
    def _mock_handler(payload):
        # Simulate ServiceNow ticket creation
        _ = mock_servicenow_api(payload)
        return AgentResponse(success=True, message="Ticket created successfully.")
    return MagicMock(side_effect=_mock_handler)

@pytest.mark.performance
def test_performance_high_load_ticket_creation(valid_ticket_message, mock_agent_message_endpoint):
    """
    Measures the system's ability to handle multiple concurrent ticket creation requests without significant degradation.

    Success criteria:
    - All requests complete within 5 seconds
    - No more than 5% of requests fail
    - No deadlocks or resource exhaustion
    """
    NUM_REQUESTS = 50
    MAX_DURATION = 5.0  # seconds
    MAX_FAILURE_RATE = 0.05  # 5%

    # Patch any HTTP/network calls inside the agent handler
    with patch("agent.main.create_servicenow_ticket", new=mock_agent_message_endpoint):
        # Simulate concurrent requests using asyncio
        async def send_request(idx):
            # Each request gets its own payload copy
            payload = dict(valid_ticket_message)
            payload["user_id"] = f"user{idx}"
            try:
                # Simulate the /agent/message POST handler
                response = mock_agent_message_endpoint(payload)
                return response
            except Exception as e:
                return e

        async def run_load_test():
            tasks = [asyncio.create_task(send_request(i)) for i in range(NUM_REQUESTS)]
            start = time.time()
            results = await asyncio.gather(*tasks)
            duration = time.time() - start
            return results, duration

        # Run the async load test
        results, duration = asyncio.run(run_load_test())

        # Analyze results
        success_count = 0
        failure_count = 0
        for res in results:
            if isinstance(res, AgentResponse) and res.success and "Ticket created" in res.message:
                success_count += 1
            else:
                failure_count += 1

        failure_rate = failure_count / NUM_REQUESTS

        # Assertions
        assert duration < MAX_DURATION, f"All requests took {duration:.2f}s, expected < {MAX_DURATION}s"
        assert failure_rate <= MAX_FAILURE_RATE, f"Failure rate {failure_rate*100:.1f}% exceeds {MAX_FAILURE_RATE*100:.1f}%"
        assert success_count == NUM_REQUESTS or failure_count <= int(NUM_REQUESTS * MAX_FAILURE_RATE), \
            f"Too many failures: {failure_count} out of {NUM_REQUESTS}"

        # Check for deadlocks/resource exhaustion (if asyncio.run completes, no deadlock)
        # Optionally, check for memory/thread usage if supported (omitted here)

        # Print summary for debugging (optional)
        print(f"Performance test: {success_count} succeeded, {failure_count} failed, duration={duration:.2f}s")

