from typing import Any, Dict, Generator, List, Optional

from .base.base_agent import BaseAgent
from .factory import AgentFactory


class AgentManager:
    """
    Manager class for LLM Agent instances.
    """

    def __init__(self):
        """
        Initialize the agent manager.
        """
        self.agents: Dict[str, BaseAgent] = {}
        self.current_agent_type: Optional[str] = None

    def initialize_agent(
        self, agent_type: str, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize an agent instance.

        Args:
            agent_type: The type of agent to initialize.
            config: Optional configuration for the agent.

        Raises:
            ValueError: If the agent type is not registered.
        """
        if agent_type not in self.agents:
            self.agents[agent_type] = AgentFactory.create_agent(agent_type, config)
        else:
            if config:
                self.agents[agent_type].update_config(config)

        self.current_agent_type = agent_type

    def get_current_agent(self) -> Optional[BaseAgent]:
        """
        Get the current agent instance.

        Returns:
            The current agent instance, or None if no agent is initialized.
        """
        if not self.current_agent_type:
            return None

        return self.agents.get(self.current_agent_type)

    def process_message(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Process a message using the current agent.

        Args:
            message: The message to process.
            context: Optional conversation context.

        Returns:
            A generator that yields response chunks.

        Raises:
            ValueError: If no agent is initialized.
        """
        agent = self.get_current_agent()
        if not agent:
            raise ValueError("No agent is initialized.")

        return agent.process_message(message, context)

    def get_response(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Get a response from the current agent.

        Args:
            message: The message to process.
            context: Optional conversation context.

        Returns:
            The agent's response.

        Raises:
            ValueError: If no agent is initialized.
        """
        agent = self.get_current_agent()
        if not agent:
            raise ValueError("No agent is initialized.")

        return agent.get_response(message, context)

    def get_available_agents(self) -> Dict[str, str]:
        """
        Get a dictionary of available agent types and their descriptions.

        Returns:
            A dictionary mapping agent types to descriptions.
        """
        return AgentFactory.get_available_agents()

    def reset_all_agents(self) -> None:
        """
        Reset all agent instances.
        """
        for agent in self.agents.values():
            agent.reset()
