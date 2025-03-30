from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional


class BaseAgent(ABC):
    """
    BaseAgent is an abstract base class that defines the interface for all LLM agents.
    All agent implementations must inherit from this class and implement its abstract methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent with configuration parameters.

        Args:
            config: A dictionary containing configuration parameters for the agent.
        """
        self.config = config or {}
        self.state: Dict[str, Any] = {}
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the agent state and resources.
        This method can be overridden by subclasses to perform additional initialization.
        """
        pass

    @abstractmethod
    def process_message(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Process a user message and generate a response.

        Args:
            message: The user message to process.
            context: Optional conversation history or context.

        Returns:
            A generator that yields response chunks for streaming and finally returns
            a dictionary containing the complete response and any additional metadata.
        """
        pass

    @abstractmethod
    def get_response(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and return a complete response (non-streaming).

        Args:
            message: The user message to process.
            context: Optional conversation history or context.

        Returns:
            A dictionary containing the complete response and any additional metadata.
        """
        pass

    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the agent's configuration.

        Args:
            config_updates: A dictionary containing configuration updates.
        """
        self.config.update(config_updates)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the agent's current configuration.

        Returns:
            The agent's configuration dictionary.
        """
        return self.config

    def reset(self) -> None:
        """
        Reset the agent's state.
        """
        self.state = {}

    def get_capabilities(self) -> List[str]:
        """
        Get a list of the agent's capabilities.

        Returns:
            A list of capability strings.
        """
        return []
