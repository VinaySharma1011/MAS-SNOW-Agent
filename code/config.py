
# config.py

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Configuration management for ServiceNow Ticketing Assistant.
    Handles environment variable loading, API key management, LLM config,
    domain-specific settings, validation, error handling, and defaults.
    """

    # Required environment variables for ServiceNow API
    SERVICENOW_INSTANCE_URL = os.getenv("SERVICENOW_INSTANCE_URL")
    SERVICENOW_CLIENT_ID = os.getenv("SERVICENOW_CLIENT_ID")
    SERVICENOW_CLIENT_SECRET = os.getenv("SERVICENOW_CLIENT_SECRET")
    SERVICENOW_USERNAME = os.getenv("SERVICENOW_USERNAME")
    SERVICENOW_PASSWORD = os.getenv("SERVICENOW_PASSWORD")

    # Required environment variables for Azure AI Search (RAG)
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
    AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

    # LLM Configuration
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4.1"
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    LLM_SYSTEM_PROMPT = (
        "You are a professional IT service desk assistant specializing in ServiceNow ticket management. "
        "Your primary responsibilities are to: 1. Create new ServiceNow tickets for users based on their requests, "
        "ensuring all required information is collected and validated. 2. Retrieve and provide the current status of "
        "ServiceNow tickets when users supply a valid ticket ID. 3. Answer general queries using the provided knowledge base via Azure AI Search. "
        "Instructions: - Always confirm the required fields (short description, category, priority) before creating a ticket. "
        "- For status requests, validate the ticket ID before proceeding. - Communicate in a formal, concise, and professional manner. "
        "- If information is not found in the knowledge base, politely inform the user and suggest next steps. - Output responses in clear, structured text."
    )
    LLM_USER_PROMPT_TEMPLATE = (
        "Please describe your issue or request a ticket status. For new tickets, include a short description, category, and priority. For status updates, provide your ticket ID."
    )
    LLM_FEW_SHOT_EXAMPLES = [
        "I need to report a network outage. -> Thank you for reporting the issue. Please provide the category and priority to proceed with ticket creation.",
        "What is the status of ticket INC123456? -> The current status of ticket INC123456 is 'In Progress'. Is there anything else I can assist you with?"
    ]

    # Domain-specific settings
    DOMAIN = "general"
    AGENT_NAME = "ServiceNow Ticketing Assistant"
    RAG_ENABLED = True
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
    RAG_SEARCH_TYPE = "vector_semantic"
    RAG_EMBEDDING_MODEL = AZURE_OPENAI_EMBEDDING_DEPLOYMENT

    # Default values and fallbacks
    DEFAULT_PRIORITY = "Medium"
    DEFAULT_CATEGORY = "General"
    FALLBACK_RESPONSE = (
        "I'm sorry, I could not find the information you requested in the knowledge base. "
        "Please provide more details or contact your IT support team for further assistance."
    )

    @classmethod
    def validate_servicenow(cls):
        missing = []
        if not cls.SERVICENOW_INSTANCE_URL:
            missing.append("SERVICENOW_INSTANCE_URL")
        if not cls.SERVICENOW_CLIENT_ID:
            missing.append("SERVICENOW_CLIENT_ID")
        if not cls.SERVICENOW_CLIENT_SECRET:
            missing.append("SERVICENOW_CLIENT_SECRET")
        if not cls.SERVICENOW_USERNAME:
            missing.append("SERVICENOW_USERNAME")
        if not cls.SERVICENOW_PASSWORD:
            missing.append("SERVICENOW_PASSWORD")
        if missing:
            raise ConfigError(f"Missing ServiceNow API configuration: {', '.join(missing)}")

    @classmethod
    def validate_azure_search(cls):
        missing = []
        if not cls.AZURE_SEARCH_ENDPOINT:
            missing.append("AZURE_SEARCH_ENDPOINT")
        if not cls.AZURE_SEARCH_API_KEY:
            missing.append("AZURE_SEARCH_API_KEY")
        if not cls.AZURE_SEARCH_INDEX_NAME:
            missing.append("AZURE_SEARCH_INDEX_NAME")
        if not cls.AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not cls.AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
            missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if missing:
            raise ConfigError(f"Missing Azure AI Search configuration: {', '.join(missing)}")

    @classmethod
    def validate_llm(cls):
        if not cls.AZURE_OPENAI_API_KEY:
            raise ConfigError("Missing AZURE_OPENAI_API_KEY for LLM.")
        if not cls.AZURE_OPENAI_ENDPOINT:
            raise ConfigError("Missing AZURE_OPENAI_ENDPOINT for LLM.")

    @classmethod
    def validate_all(cls):
        try:
            cls.validate_servicenow()
            cls.validate_azure_search()
            cls.validate_llm()
        except ConfigError as e:
            logging.error(str(e))
            raise

    @classmethod
    def get_servicenow_credentials(cls):
        cls.validate_servicenow()
        return {
            "instance_url": cls.SERVICENOW_INSTANCE_URL,
            "client_id": cls.SERVICENOW_CLIENT_ID,
            "client_secret": cls.SERVICENOW_CLIENT_SECRET,
            "username": cls.SERVICENOW_USERNAME,
            "password": cls.SERVICENOW_PASSWORD
        }

    @classmethod
    def get_azure_search_config(cls):
        cls.validate_azure_search()
        return {
            "endpoint": cls.AZURE_SEARCH_ENDPOINT,
            "api_key": cls.AZURE_SEARCH_API_KEY,
            "index_name": cls.AZURE_SEARCH_INDEX_NAME,
            "openai_endpoint": cls.AZURE_OPENAI_ENDPOINT,
            "openai_api_key": cls.AZURE_OPENAI_API_KEY,
            "embedding_deployment": cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        }

    @classmethod
    def get_llm_config(cls):
        cls.validate_llm()
        return {
            "provider": cls.LLM_PROVIDER,
            "model": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "system_prompt": cls.LLM_SYSTEM_PROMPT,
            "user_prompt_template": cls.LLM_USER_PROMPT_TEMPLATE,
            "few_shot_examples": cls.LLM_FEW_SHOT_EXAMPLES
        }

    @classmethod
    def get_domain_settings(cls):
        return {
            "domain": cls.DOMAIN,
            "agent_name": cls.AGENT_NAME,
            "rag_enabled": cls.RAG_ENABLED,
            "rag_top_k": cls.RAG_TOP_K,
            "rag_search_type": cls.RAG_SEARCH_TYPE,
            "rag_embedding_model": cls.RAG_EMBEDDING_MODEL,
            "default_priority": cls.DEFAULT_PRIORITY,
            "default_category": cls.DEFAULT_CATEGORY,
            "fallback_response": cls.FALLBACK_RESPONSE
        }

# Error handling for missing API keys and critical config
try:
    Config.validate_all()
except ConfigError as e:
    logging.error(f"Configuration error: {e}")
    # Optionally: raise or exit depending on deployment context
    # raise

