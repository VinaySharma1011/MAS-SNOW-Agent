try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import logging
import re
import time as _time
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field, model_validator
from dotenv import load_dotenv

import openai
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Observability wrappers (trace_step, trace_step_sync, etc.) are injected by the runtime.
# Do NOT import or define them here.

# Load .env if present
load_dotenv()

# Logging configuration
logger = logging.getLogger("servicenow_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
)
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# ------------------ CONFIGURATION ------------------

class Config:
    """Centralized configuration management for environment variables."""

    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(key, default)

    @staticmethod
    def validate_servicenow():
        missing = []
        if not Config.get("SERVICENOW_INSTANCE_URL"):
            missing.append("SERVICENOW_INSTANCE_URL")
        if not Config.get("SERVICENOW_CLIENT_ID"):
            missing.append("SERVICENOW_CLIENT_ID")
        if not Config.get("SERVICENOW_CLIENT_SECRET"):
            missing.append("SERVICENOW_CLIENT_SECRET")
        if not Config.get("SERVICENOW_USERNAME"):
            missing.append("SERVICENOW_USERNAME")
        if not Config.get("SERVICENOW_PASSWORD"):
            missing.append("SERVICENOW_PASSWORD")
        if missing:
            raise RuntimeError(f"Missing ServiceNow config: {', '.join(missing)}")

    @staticmethod
    def validate_azure_search():
        missing = []
        for k in [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        ]:
            if not Config.get(k):
                missing.append(k)
        if missing:
            raise RuntimeError(f"Missing Azure Search config: {', '.join(missing)}")

    @staticmethod
    def validate_openai():
        if not Config.get("AZURE_OPENAI_API_KEY"):
            raise RuntimeError("Missing AZURE_OPENAI_API_KEY")
        if not Config.get("AZURE_OPENAI_ENDPOINT"):
            raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT")

# ------------------ DOMAIN MODELS ------------------

class ParsedRequest(BaseModel):
    """Represents a parsed user request."""
    raw_text: str
    cleaned_text: str
    entities: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("raw_text")
    @classmethod
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty.")
        if len(v) > 50000:
            raise ValueError("Input text exceeds 50,000 characters.")
        return v.strip()

    @field_validator("cleaned_text")
    @classmethod
    def clean_text(cls, v):
        return v.strip() if v else ""

class Intent(str):
    """Intent types."""
    CREATE_TICKET = "create_ticket"
    GET_STATUS = "get_status"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    error_code: Optional[str] = None

class TicketResponse(BaseModel):
    success: bool
    ticket_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    success: bool
    ticket_id: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class DocumentChunk(BaseModel):
    chunk: str
    title: Optional[str] = None
    source: Optional[str] = None

class RuleResult(BaseModel):
    passed: bool
    errors: List[str] = Field(default_factory=list)
    error_code: Optional[str] = None

# ------------------ PRESENTATION LAYER ------------------

class UserInputHandler:
    """Receives and parses user messages, detects intent, and validates input format."""

    def __init__(self, intent_classifier, input_validator):
        self.intent_classifier = intent_classifier
        self.input_validator = input_validator

    async def handle_input(self, user_message: str) -> ParsedRequest:
        """Parse and clean user input."""
        async with trace_step(
            "parse_input", step_type="parse",
            decision_summary="Parse and clean user input",
            output_fn=lambda r: f"entities={r.entities}"
        ) as step:
            cleaned = user_message.strip()
            entities = self._extract_entities(cleaned)
            parsed = ParsedRequest(raw_text=user_message, cleaned_text=cleaned, entities=entities)
            step.capture(parsed)
            return parsed

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Simple entity extraction for ticket fields and IDs."""
        entities = {}
        # Extract ticket ID (e.g., INC123456)
        ticket_id_match = re.search(r"\bINC\d{6,}\b", text, re.IGNORECASE)
        if ticket_id_match:
            entities["ticket_id"] = ticket_id_match.group(0).upper()
        # Extract possible fields (very basic, can be improved)
        for field in ["short description", "category", "priority", "impact", "urgency"]:
            pattern = rf"{field}[:=]\s*([^\n,;]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities[field.replace(" ", "_")] = match.group(1).strip()
        return entities

# ------------------ APPLICATION LAYER ------------------

class IntentClassifier:
    """Classifies user intent using LLM."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def classify(self, parsed_input: ParsedRequest) -> str:
        """Classify intent using LLM."""
        async with trace_step(
            "classify_intent", step_type="llm_call",
            decision_summary="Classify user intent using LLM",
            output_fn=lambda r: f"intent={r}"
        ) as step:
            try:
                prompt = (
                    "Classify the user's intent as one of: create_ticket, get_status, general_query.\n"
                    "User message: " + parsed_input.cleaned_text +
                    "\nIf the user is asking to create a ticket, respond with 'create_ticket'. "
                    "If asking for ticket status (with a ticket ID), respond with 'get_status'. "
                    "If asking a general question, respond with 'general_query'. "
                    "If unclear, respond with 'unknown'."
                )
                response = await self.llm_client.generate_response(prompt, {})
                intent = response.strip().splitlines()[0].lower()
                if "create_ticket" in intent:
                    result = Intent.CREATE_TICKET
                elif "get_status" in intent:
                    result = Intent.GET_STATUS
                elif "general_query" in intent:
                    result = Intent.GENERAL_QUERY
                else:
                    result = Intent.UNKNOWN
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Intent classification failed: {e}")
                step.capture(Intent.UNKNOWN)
                return Intent.UNKNOWN

# ------------------ DOMAIN LAYER ------------------

class InputValidator:
    """Validates required fields for ticket creation and status retrieval."""

    def __init__(self, business_rules_engine):
        self.business_rules_engine = business_rules_engine

    def validate(self, intent: str, data: dict) -> ValidationResult:
        """Validate input fields according to business rules."""
        with trace_step_sync(
            "validate_input", step_type="process",
            decision_summary="Validate input fields for intent",
            output_fn=lambda r: f"is_valid={r.is_valid}"
        ) as step:
            result = self.business_rules_engine.apply_rules(intent, data)
            step.capture(result)
            return ValidationResult(
                is_valid=result.passed,
                errors=result.errors,
                error_code=result.error_code
            )

class BusinessRulesEngine:
    """Applies business rules for validation, mapping, and decision tables."""

    def apply_rules(self, intent: str, data: dict) -> RuleResult:
        """Apply business rules for the given intent and data."""
        errors = []
        error_code = None
        if intent == Intent.CREATE_TICKET:
            # Required: short_description, category, priority
            for field in ["short_description", "category", "priority"]:
                if not data.get(field):
                    errors.append(f"Missing required field: {field}")
            if errors:
                error_code = "SNOW_INVALID_INPUT"
                return RuleResult(passed=False, errors=errors, error_code=error_code)
            return RuleResult(passed=True)
        elif intent == Intent.GET_STATUS:
            if not data.get("ticket_id"):
                errors.append("Missing ticket_id for status retrieval.")
                error_code = "SNOW_INVALID_INPUT"
                return RuleResult(passed=False, errors=errors, error_code=error_code)
            return RuleResult(passed=True)
        # For general queries, no validation needed
        return RuleResult(passed=True)

    def map_fields(self, data: dict) -> dict:
        """Map and transform user input to ServiceNow API schema."""
        mapped = {}
        if "short_description" in data:
            mapped["short_description"] = data["short_description"].strip()
        if "category" in data:
            mapped["category"] = data["category"].capitalize()
        if "priority" in data:
            mapped["priority"] = data["priority"].upper()
        # Optionally map impact/urgency to priority using decision table
        if "impact" in data and "urgency" in data:
            mapped["priority"] = self._priority_decision_table(
                data.get("impact"), data.get("urgency")
            )
        return mapped

    def _priority_decision_table(self, impact: str, urgency: str) -> str:
        """Apply decision table for priority assignment."""
        if not impact or not urgency:
            return "Medium"
        impact = impact.lower()
        urgency = urgency.lower()
        if impact == "high" and urgency == "high":
            return "Critical"
        if impact == "medium" and urgency == "high":
            return "High"
        if impact == "low" and urgency == "low":
            return "Low"
        return "Medium"

# ------------------ INTEGRATION LAYER ------------------

class OAuth2Authenticator:
    """Manages OAuth2 authentication for ServiceNow API access."""

    def __init__(self):
        self.token = None
        self.token_expiry = 0

    def get_token(self) -> str:
        """Obtain OAuth2 token for ServiceNow API."""
        now = int(_time.time())
        if self.token and now < self.token_expiry - 60:
            return self.token
        Config.validate_servicenow()
        url = f"{Config.get('SERVICENOW_INSTANCE_URL').rstrip('/')}/oauth_token.do"
        client_id = Config.get("SERVICENOW_CLIENT_ID")
        client_secret = Config.get("SERVICENOW_CLIENT_SECRET")
        username = Config.get("SERVICENOW_USERNAME")
        password = Config.get("SERVICENOW_PASSWORD")
        data = {
            "grant_type": "password",
            "client_id": client_id,
            "client_secret": client_secret,
            "username": username,
            "password": password,
        }
        try:
            _obs_t0 = _time.time()
            resp = requests.post(url, data=data, timeout=10)
            try:
                trace_tool_call(
                    tool_name='requests.post',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(resp)[:200] if resp is not None else None,
                    status="success",
                )
            except Exception:
                pass
            resp.raise_for_status()
            result = resp.json()
            self.token = result["access_token"]
            self.token_expiry = now + int(result.get("expires_in", 1800))
            return self.token
        except Exception as e:
            logger.error(f"OAuth2 token retrieval failed: {e}")
            raise

class ServiceNowAPIClient:
    """Handles authenticated communication with ServiceNow APIs."""

    def __init__(self, authenticator: OAuth2Authenticator):
        self.authenticator = authenticator

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True
    )
    def create_ticket(self, ticket_data: dict) -> TicketResponse:
        """Create a new ServiceNow ticket."""
        with trace_step_sync(
            "create_servicenow_ticket", step_type="tool_call",
            decision_summary="Create ServiceNow ticket via API",
            output_fn=lambda r: f"ticket_id={r.ticket_id if r.ticket_id else '?'}"
        ) as step:
            Config.validate_servicenow()
            url = f"{Config.get('SERVICENOW_INSTANCE_URL').rstrip('/')}/api/now/table/incident"
            token = self.authenticator.get_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {
                "short_description": ticket_data.get("short_description"),
                "category": ticket_data.get("category"),
                "priority": ticket_data.get("priority"),
            }
            try:
                _obs_t0 = _time.time()
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                try:
                    trace_tool_call(
                        tool_name='requests.post',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                resp.raise_for_status()
                result = resp.json()
                ticket_id = result.get("result", {}).get("number")
                message = f"Ticket {ticket_id} created successfully."
                response = TicketResponse(success=True, ticket_id=ticket_id, message=message)
                step.capture(response)
                return response
            except Exception as e:
                logger.error(f"ServiceNow ticket creation failed: {e}")
                response = TicketResponse(success=False, error=str(e))
                step.capture(response)
                return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True
    )
    def get_ticket_status(self, ticket_id: str) -> StatusResponse:
        """Retrieve the status of an existing ServiceNow ticket."""
        with trace_step_sync(
            "get_ticket_status", step_type="tool_call",
            decision_summary="Retrieve ServiceNow ticket status",
            output_fn=lambda r: f"status={r.status if r.status else '?'}"
        ) as step:
            Config.validate_servicenow()
            url = f"{Config.get('SERVICENOW_INSTANCE_URL').rstrip('/')}/api/now/table/incident"
            token = self.authenticator.get_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json"
            }
            params = {
                "sysparm_query": f"number={ticket_id}",
                "sysparm_limit": 1
            }
            try:
                _obs_t0 = _time.time()
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                try:
                    trace_tool_call(
                        tool_name='requests.get',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                resp.raise_for_status()
                result = resp.json()
                records = result.get("result", [])
                if not records:
                    response = StatusResponse(success=False, ticket_id=ticket_id, error="Ticket not found.")
                    step.capture(response)
                    return response
                status = records[0].get("state", "Unknown")
                message = f"The current status of ticket {ticket_id} is '{status}'."
                response = StatusResponse(success=True, ticket_id=ticket_id, status=status, message=message)
                step.capture(response)
                return response
            except Exception as e:
                logger.error(f"ServiceNow ticket status retrieval failed: {e}")
                response = StatusResponse(success=False, ticket_id=ticket_id, error=str(e))
                step.capture(response)
                return response

class AzureAISearchClient:
    """Queries Azure AI Search for knowledge base answers."""

    def __init__(self):
        self._search_client = None
        self._openai_client = None

    def _get_search_client(self):
        if self._search_client is None:
            Config.validate_azure_search()
            self._search_client = SearchClient(
                endpoint=Config.get("AZURE_SEARCH_ENDPOINT"),
                index_name=Config.get("AZURE_SEARCH_INDEX_NAME"),
                credential=AzureKeyCredential(Config.get("AZURE_SEARCH_API_KEY")),
            )
        return self._search_client

    def _get_openai_client(self):
        if self._openai_client is None:
            Config.validate_azure_search()
            self._openai_client = openai.AzureOpenAI(
                api_key=Config.get("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=Config.get("AZURE_OPENAI_ENDPOINT"),
            )
        return self._openai_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Search Azure AI Search index for relevant knowledge base chunks."""
        with trace_step_sync(
            "search_knowledge_base", step_type="tool_call",
            decision_summary="Search Azure AI Search for knowledge base answers",
            output_fn=lambda r: f"chunks={len(r)}"
        ) as step:
            search_client = self._get_search_client()
            openai_client = self._get_openai_client()
            embedding_resp = openai_client.embeddings.create(
                input=query,
                model=Config.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
            )
            vector_query = VectorizedQuery(
                vector=embedding_resp.data[0].embedding,
                k_nearest_neighbors=top_k,
                fields="vector"
            )
            results = search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=top_k,
                select=["chunk", "title"]
            )
            chunks = []
            for r in results:
                if r.get("chunk"):
                    chunks.append(DocumentChunk(chunk=r["chunk"], title=r.get("title")))
            step.capture(chunks)
            return chunks

# ------------------ LLM CLIENT ------------------

class LLMClient:
    """Handles prompt construction and LLM calls."""

    def __init__(self, model: str, temperature: float, max_tokens: int, system_prompt: str):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._openai_client = None

    def _get_client(self):
        if self._openai_client is None:
            Config.validate_azure_search()
            self._openai_client = openai.AsyncOpenAI(
                api_key=Config.get("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=Config.get("AZURE_OPENAI_ENDPOINT"),
            )
        return self._openai_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_response(self, prompt: str, context: dict) -> str:
        """Call LLM to generate a response."""
        async with trace_step(
            "generate_response", step_type="llm_call",
            decision_summary="Call LLM to produce a reply",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            # Optionally add context as assistant message
            if context and context.get("knowledge_chunks"):
                context_text = "\n".join([c.chunk for c in context["knowledge_chunks"]])
                messages.append({"role": "assistant", "content": context_text})
            _t0 = _time.time()
            response = await self._get_client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="openai", model_name=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else ""
                )
            except Exception:
                pass
            step.capture(content)
            return content

# ------------------ RESPONSE FORMATTER ------------------

class ResponseFormatter:
    """Formats agent responses according to output templates."""

    def format_response(self, response_data: dict, template: str) -> str:
        """Format response using template."""
        with trace_step_sync(
            "format_response", step_type="format",
            decision_summary="Format agent response",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            try:
                # Simple template rendering
                if "message" in response_data and response_data["message"]:
                    result = response_data["message"]
                elif "error" in response_data and response_data["error"]:
                    result = f"Error: {response_data['error']}"
                else:
                    result = template
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Response formatting failed: {e}")
                fallback = "An error occurred while formatting the response."
                step.capture(fallback)
                return fallback

# ------------------ ERROR HANDLER ------------------

class ErrorHandler:
    """Handles errors, retries, escalation, and user-facing error messages."""

    def __init__(self, logger):
        self.logger = logger

    def handle_error(self, error: Exception, context: dict = None) -> str:
        """Return user-friendly error message and log the error."""
        self.logger.log(
            event="error",
            level="error",
            data={"error": str(error), "context": context or {}}
        )
        return (
            "An unexpected error occurred. Please try again later or contact IT support if the issue persists."
        )

# ------------------ LOGGER ------------------

class Logger:
    """Logs events, errors, and audit trails."""

    def log(self, event: str, level: str, data: dict) -> None:
        msg = f"{event.upper()} | {data}"
        if level.lower() == "error":
            logger.error(msg)
        elif level.lower() == "warning":
            logger.warning(msg)
        else:
            logger.info(msg)

# ------------------ MAIN AGENT ------------------

class ServiceNowTicketingAssistant:
    """Main agent class orchestrating all components."""

    def __init__(self):
        # Compose all components
        self.logger = Logger()
        self.error_handler = ErrorHandler(self.logger)
        self.business_rules_engine = BusinessRulesEngine()
        self.input_validator = InputValidator(self.business_rules_engine)
        self.oauth2_authenticator = OAuth2Authenticator()
        self.servicenow_api_client = ServiceNowAPIClient(self.oauth2_authenticator)
        self.azure_ai_search_client = AzureAISearchClient()
        self.llm_client = LLMClient(
            model="gpt-4.1",
            temperature=0.7,
            max_tokens=2000,
            system_prompt=(
                "You are a professional IT service desk assistant specializing in ServiceNow ticket management. "
                "Your primary responsibilities are to: 1. Create new ServiceNow tickets for users based on their requests, "
                "ensuring all required information is collected and validated. 2. Retrieve and provide the current status of "
                "ServiceNow tickets when users supply a valid ticket ID. 3. Answer general queries using the provided knowledge base via Azure AI Search. "
                "Instructions: - Always confirm the required fields (short description, category, priority) before creating a ticket. "
                "- For status requests, validate the ticket ID before proceeding. - Communicate in a formal, concise, and professional manner. "
                "- If information is not found in the knowledge base, politely inform the user and suggest next steps. - Output responses in clear, structured text."
            )
        )
        self.intent_classifier = IntentClassifier(self.llm_client)
        self.user_input_handler = UserInputHandler(self.intent_classifier, self.input_validator)
        self.response_formatter = ResponseFormatter()
        self.fallback_response = (
            "I'm sorry, I could not find the information you requested in the knowledge base. "
            "Please provide more details or contact your IT support team for further assistance."
        )

    @trace_agent(agent_name='ServiceNow Ticketing Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def handle_user_message(self, user_message: str) -> str:
        """Entry point for processing user messages."""
        async with trace_step(
            "handle_user_message", step_type="plan",
            decision_summary="Orchestrate intent detection, validation, and response generation",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            try:
                parsed = await self.user_input_handler.handle_input(user_message)
                intent = await self.classify_intent(parsed)
                data = parsed.entities.copy()
                # For ticket creation, try to extract fields from text if not present
                if intent == Intent.CREATE_TICKET:
                    # Try to extract fields from text if missing
                    for field in ["short_description", "category", "priority"]:
                        if not data.get(field):
                            # Try to extract from text using regex
                            match = re.search(rf"{field.replace('_', ' ')}[:=]\s*([^\n,;]+)", parsed.cleaned_text, re.IGNORECASE)
                            if match:
                                data[field] = match.group(1).strip()
                    # If still missing, try to use the whole text as short_description
                    if not data.get("short_description"):
                        data["short_description"] = parsed.cleaned_text
                validation = self.validate_input(intent, data)
                if not validation.is_valid:
                    response = {
                        "success": False,
                        "error": "; ".join(validation.errors),
                        "error_code": validation.error_code
                    }
                    formatted = self.response_formatter.format_response(response, self.fallback_response)
                    step.capture(formatted)
                    return formatted
                # Map fields as per business rules
                mapped_data = self.business_rules_engine.map_fields(data)
                # Route to appropriate handler
                if intent == Intent.CREATE_TICKET:
                    ticket_response = self.create_servicenow_ticket(mapped_data)
                    formatted = self.response_formatter.format_response(ticket_response.model_dump(), self.fallback_response)
                    step.capture(formatted)
                    return formatted
                elif intent == Intent.GET_STATUS:
                    ticket_id = data.get("ticket_id")
                    status_response = self.get_ticket_status(ticket_id)
                    formatted = self.response_formatter.format_response(status_response.model_dump(), self.fallback_response)
                    step.capture(formatted)
                    return formatted
                elif intent == Intent.GENERAL_QUERY:
                    # RAG pipeline: retrieve context from Azure AI Search
                    chunks = self.search_knowledge_base(parsed.cleaned_text, top_k=5)
                    if not chunks:
                        step.capture(self.fallback_response)
                        return self.fallback_response
                    # Pass retrieved chunks as context to LLM
                    context = {"knowledge_chunks": chunks}
                    answer = await self.llm_client.generate_response(parsed.cleaned_text, context)
                    if not answer or "I could not find" in answer:
                        step.capture(self.fallback_response)
                        return self.fallback_response
                    step.capture(answer)
                    return answer
                else:
                    step.capture(self.fallback_response)
                    return self.fallback_response
            except Exception as e:
                error_msg = self.error_handler.handle_error(e, {"user_message": user_message})
                step.capture(error_msg)
                return error_msg

    async def classify_intent(self, parsed_input: ParsedRequest) -> str:
        """Classifies the user's intent using LLM."""
        return await self.intent_classifier.classify(parsed_input)

    def validate_input(self, intent: str, data: dict) -> ValidationResult:
        """Validates user input fields according to business rules."""
        return self.input_validator.validate(intent, data)

    def create_servicenow_ticket(self, ticket_data: dict) -> TicketResponse:
        """Creates a new ServiceNow ticket via API after validation."""
        return self.servicenow_api_client.create_ticket(ticket_data)

    def get_ticket_status(self, ticket_id: str) -> StatusResponse:
        """Retrieves the status of an existing ServiceNow ticket."""
        return self.servicenow_api_client.get_ticket_status(ticket_id)

    @trace_agent(agent_name='ServiceNow Ticketing Assistant')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Queries Azure AI Search for knowledge base answers."""
        return self.azure_ai_search_client.search_knowledge_base(query, top_k)

    def format_response(self, response_data: dict, template: str) -> str:
        """Formats the final response for the user using templates."""
        return self.response_formatter.format_response(response_data, template)

# ------------------ FASTAPI PRESENTATION LAYER ------------------

class UserMessageRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=50000)

    @field_validator("user_message")
    @classmethod
    def clean_and_validate(cls, v):
        if not v or not v.strip():
            raise ValueError("User message cannot be empty.")
        return v.strip()

class AgentResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    tips: Optional[str] = None

# FastAPI app
app = FastAPI(
    title="ServiceNow Ticketing Assistant",
    description="Professional IT service desk assistant for ServiceNow ticket management.",
    version="1.0.0"
)

# CORS (allow all origins for demo; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ServiceNowTicketingAssistant()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Malformed input. Please check your request format.",
            "error_type": "validation_error",
            "tips": "Ensure your JSON is valid and all required fields are present."
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_type": "http_error",
            "tips": "Check your request and try again."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error.",
            "error_type": "internal_error",
            "tips": "Please try again later or contact support."
        }
    )

@app.post("/agent/message", response_model=AgentResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def agent_message(request: UserMessageRequest):
    """Main endpoint for user messages."""
    try:
        response_text = await agent.handle_user_message(request.user_message)
        return AgentResponse(success=True, response=response_text)
    except ValidationError as ve:
        logger.warning(f"Input validation error: {ve}")
        return AgentResponse(
            success=False,
            error="Invalid input.",
            error_type="validation_error",
            tips="Ensure your message is not empty and within 50,000 characters."
        )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return AgentResponse(
            success=False,
            error="An error occurred while processing your request.",
            error_type="agent_error",
            tips="Please try again later or contact IT support."
        )

@app.get("/health")
async def health_check():
    return {"success": True, "status": "ok"}

# ------------------ MAIN ENTRY POINT ------------------



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting ServiceNow Ticketing Assistant agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())