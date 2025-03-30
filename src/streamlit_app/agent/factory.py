from typing import Any, Dict, Optional, Type

from .base.base_agent import BaseAgent
from .implementations.azure_openai_agent import AzureOpenAIAgent
from .implementations.context_aware_agent import ContextAwareAgent
from .implementations.tool_using_agent import ToolUsingAgent


class AgentFactory:
    """
    Factory class for creating LLM Agent instances.
    """

    _agent_classes = {
        "azure_openai": AzureOpenAIAgent,
        "context_aware": ContextAwareAgent,
        "tool_using": ToolUsingAgent,
    }

    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent class.

        Args:
            name: The name of the agent.
            agent_class: The agent class to register.
        """
        cls._agent_classes[name] = agent_class

    @classmethod
    def create_agent(
        cls, agent_type: str, config: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """
        Create an agent instance.

        Args:
            agent_type: The type of agent to create.
            config: Optional configuration for the agent.

        Returns:
            An instance of the specified agent type.

        Raises:
            ValueError: If the agent type is not registered.
        """
        if agent_type not in cls._agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = cls._agent_classes[agent_type]
        return agent_class(config)

    @classmethod
    def get_available_agents(cls) -> Dict[str, str]:
        """
        Get a dictionary of available agent types and their descriptions.

        Returns:
            A dictionary mapping agent types to descriptions.
        """
        return {
            "azure_openai": "Azure OpenAI APIを直接使用する基本エージェント",
            "context_aware": "ドキュメントコンテキストを活用した情報提供エージェント",
            "tool_using": "外部ツール/APIを利用できる拡張エージェント",
        }
