"""
LLM Provider Configuration for Multiple Agents.

Simplified structure matching .env format:
- [AGENT]_PROVIDER: "openrouter" or "openai"
- [AGENT]_API_KEY: API key for the selected provider
- [AGENT]_MODEL: Model identifier
- [AGENT]_TEMPERATURE: (optional) Temperature setting

Each agent can use different providers and models by changing .env only.
"""

import os
import logging
from typing import Literal, Optional
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Pre-configure environment variables ONCE for OpenRouter and OpenAI
# Find the first valid API key for each provider type
openrouter_key_set = False
openai_key_set = False

for agent in ["EXTRACTOR", "CLASSIFIER", "AGGREGATOR"]:
    provider = os.getenv(f"{agent}_PROVIDER", "openrouter").lower()
    api_key = os.getenv(f"{agent}_API_KEY")
    
    if api_key and not api_key.startswith("sk-or-your"):  # Skip placeholder keys
        if provider == "openrouter" and not openrouter_key_set:
            os.environ["OPENROUTER_API_KEY"] = api_key
            openrouter_key_set = True
        elif provider == "openai" and not openai_key_set:
            os.environ["OPENAI_API_KEY"] = api_key
            openai_key_set = True

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Available LLM providers."""
    OPENROUTER = "openrouter"
    OPENAI = "openai"


@dataclass
class AgentModelConfig:
    """Configuration for a single agent's LLM."""
    
    agent_name: str
    provider: Provider
    model_name: str
    api_key: str
    temperature: float
    base_url: str


class ProviderManager:
    """
    Manages LLM provider configuration for all agents.
    
    Loads from simplified .env format:
    - EXTRACTOR_PROVIDER=openrouter
    - EXTRACTOR_API_KEY=sk-or-...
    - EXTRACTOR_MODEL=deepseek/deepseek-chat-v3.1:free
    - EXTRACTOR_TEMPERATURE=0.1 (optional)
    """
    
    # Default temperature settings per agent if not specified
    DEFAULT_TEMPERATURES = {
        "extractor": 0.1,    # Low for consistent extraction
        "classifier": 0.2,   # Slightly higher for reasoning
        "aggregator": 0.1    # Low for pattern matching
    }
    
    # Base URLs for providers
    PROVIDER_BASE_URLS = {
        Provider.OPENROUTER: "https://openrouter.ai/api/v1",
        Provider.OPENAI: "https://api.openai.com/v1"
    }
    
    def __init__(self):
        """Initialize provider manager."""
        self._configs = {}
    
    def get_agent_config(self, agent_name: str) -> AgentModelConfig:
        """
        Get LLM configuration for a specific agent.
        
        Args:
            agent_name: "extractor", "classifier", or "aggregator"
            
        Returns:
            AgentModelConfig with provider settings
            
        Raises:
            ValueError: If configuration is invalid or missing
        """
        # Cache config if already loaded
        if agent_name in self._configs:
            return self._configs[agent_name]
        
        agent_upper = agent_name.upper()
        
        # Get provider choice
        provider_str = os.getenv(f"{agent_upper}_PROVIDER", "openrouter").lower()
        
        try:
            provider = Provider(provider_str)
        except ValueError:
            raise ValueError(
                f"Invalid {agent_upper}_PROVIDER: '{provider_str}'. "
                f"Must be 'openrouter' or 'openai'"
            )
        
        # Get API key (unified for both providers per agent)
        api_key = os.getenv(f"{agent_upper}_API_KEY")
        if not api_key:
            raise ValueError(
                f"Missing {agent_upper}_API_KEY in .env file. "
                f"Get key at: https://openrouter.ai/ or https://platform.openai.com/"
            )
        
        # Get model name (unified for both providers per agent)
        model_name = os.getenv(f"{agent_upper}_MODEL")
        if not model_name:
            # Set sensible defaults based on provider
            if provider == Provider.OPENROUTER:
                model_name = "deepseek/deepseek-chat-v3.1:free"
            else:  # openai
                model_name = "gpt-4o-mini" if agent_name == "extractor" else "gpt-4o"
        
        # Get temperature (use default if not set)
        temperature_str = os.getenv(f"{agent_upper}_TEMPERATURE")
        if temperature_str:
            temperature = float(temperature_str)
        else:
            temperature = self.DEFAULT_TEMPERATURES.get(agent_name, 0.1)
        
        # Get base URL for provider
        base_url = self.PROVIDER_BASE_URLS[provider]
        
        # Create config
        config = AgentModelConfig(
            agent_name=agent_name,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            base_url=base_url
        )
        
        logger.info(
            f"Configured {agent_name}: provider={provider.value}, "
            f"model={model_name}, temperature={temperature}"
        )
        
        # Cache config
        self._configs[agent_name] = config
        return config
    
    def get_model(self, agent_name: str):
        """
        Get initialized LLM model instance for an agent.
        
        Args:
            agent_name: "extractor", "classifier", or "aggregator"
            
        Returns:
            Pydantic AI compatible model instance
        """
        config = self.get_agent_config(agent_name)
        return self._create_model_instance(config)
    
    @staticmethod
    def _create_model_instance(config: AgentModelConfig):
        """
        Create Pydantic AI model instance from config.
        
        For pydantic-ai 0.3.2:
        - OpenRouter: Pass provider='openrouter' and model name
        - OpenAI: Pass provider='openai' (or omit) and model name
        - API keys set via environment variables (OPENROUTER_API_KEY or OPENAI_API_KEY)
        """
        from pydantic_ai.models.openai import OpenAIModel
        
        if config.provider == Provider.OPENROUTER:
            # For OpenRouter, use provider parameter
            return OpenAIModel(
                config.model_name,
                provider='openrouter'
            )
        else:  # openai
            # For OpenAI, use provider parameter or default
            return OpenAIModel(
                config.model_name,
                provider='openai'
            )
    
    def get_model_info(self, agent_name: str) -> dict:
        """
        Get human-readable info about agent's configured model.
        
        Args:
            agent_name: "extractor", "classifier", or "aggregator"
            
        Returns:
            Dictionary with model information
        """
        config = self.get_agent_config(agent_name)
        
        # Determine cost based on provider and model
        if config.provider == Provider.OPENROUTER:
            cost = "FREE" if "free" in config.model_name.lower() else "PAID (OpenRouter)"
        else:  # openai
            if "gpt-4o-mini" in config.model_name:
                cost = "~$0.15/1M tokens"
            elif "gpt-4o" in config.model_name:
                cost = "~$2.50/1M tokens"
            else:
                cost = "PAID (OpenAI)"
        
        return {
            "agent": agent_name,
            "provider": config.provider.value,
            "model": config.model_name,
            "temperature": config.temperature,
            "cost": cost,
            "base_url": config.base_url
        }
    
    def validate_all_agents(self) -> dict:
        """
        Validate configuration for all agents.
        
        Returns:
            Dictionary with validation results
        """
        agents = ["extractor", "classifier", "aggregator"]
        results = {}
        
        for agent in agents:
            try:
                config = self.get_agent_config(agent)
                results[agent] = {
                    "status": "valid",
                    "provider": config.provider.value,
                    "model": config.model_name,
                    "error": None
                }
            except Exception as e:
                results[agent] = {
                    "status": "invalid",
                    "provider": None,
                    "model": None,
                    "error": str(e)
                }
        
        return results


# Global instance - use throughout your application
provider_manager = ProviderManager()