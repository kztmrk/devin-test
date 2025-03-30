import time
from typing import Any, Dict, Generator, List, Optional

from openai import AzureOpenAI

from ..base.base_agent import BaseAgent


class AzureOpenAIAgent(BaseAgent):
    """
    An implementation of BaseAgent that uses Azure OpenAI API for generating responses.
    This agent directly communicates with Azure OpenAI API and supports streaming responses.
    """

    def initialize(self) -> None:
        """
        Initialize the Azure OpenAI client.
        """
        self.client = None
        if all(k in self.config for k in ["api_key", "api_version", "azure_endpoint"]):
            self.client = AzureOpenAI(
                api_key=self.config["api_key"],
                api_version=self.config["api_version"],
                azure_endpoint=self.config["azure_endpoint"],
            )

        if "deployment_name" not in self.config:
            self.config["deployment_name"] = "gpt-35-turbo"

        if "system_message" not in self.config:
            self.config["system_message"] = "You are a helpful assistant."

    def process_message(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Process a user message and generate a streaming response using Azure OpenAI API.

        Args:
            message: The user message to process.
            context: Optional conversation history or context.

        Returns:
            A generator that yields response chunks for streaming and finally returns
            a dictionary containing the complete response and any additional metadata.
        """
        if not self.client:
            raise ValueError(
                "Azure OpenAI client is not initialized. Please check your configuration."
            )

        messages = []

        if "system_message" in self.config:
            messages.append(
                {"role": "system", "content": self.config["system_message"]}
            )

        if context:
            messages.extend(context)

        messages.append({"role": "user", "content": message})

        try:
            stream = self.client.chat.completions.create(
                model=self.config["deployment_name"],
                messages=messages,
                stream=True,
            )

            full_response = ""

            for chunk in stream:
                if (
                    chunk.choices
                    and len(chunk.choices) > 0
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content is not None
                ):
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

            return {
                "response": full_response,
                "model": self.config["deployment_name"],
                "timestamp": time.time(),
            }

        except Exception as e:
            error_message = f"Error calling Azure OpenAI API: {str(e)}"
            yield error_message
            return {
                "response": error_message,
                "error": str(e),
                "timestamp": time.time(),
            }

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
        if not self.client:
            raise ValueError(
                "Azure OpenAI client is not initialized. Please check your configuration."
            )

        messages = []

        if "system_message" in self.config:
            messages.append(
                {"role": "system", "content": self.config["system_message"]}
            )

        if context:
            messages.extend(context)

        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.config["deployment_name"],
                messages=messages,
                stream=False,
            )

            content = response.choices[0].message.content

            return {
                "response": content,
                "model": self.config["deployment_name"],
                "timestamp": time.time(),
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        except Exception as e:
            error_message = f"Error calling Azure OpenAI API: {str(e)}"
            return {
                "response": error_message,
                "error": str(e),
                "timestamp": time.time(),
            }

    def get_capabilities(self) -> List[str]:
        """
        Get a list of the agent's capabilities.

        Returns:
            A list of capability strings.
        """
        return [
            "text_generation",
            "streaming_response",
            "conversation_context",
        ]
