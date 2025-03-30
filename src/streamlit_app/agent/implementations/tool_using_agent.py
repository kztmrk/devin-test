import re
import time
from typing import Any, Callable, Dict, Generator, List, Optional

from openai import AzureOpenAI

from ..base.base_agent import BaseAgent


class Tool:
    """
    A class representing a tool that can be used by the agent.
    """

    def __init__(self, name: str, description: str, function: Callable):
        """
        Initialize a tool.

        Args:
            name: The name of the tool.
            description: A description of what the tool does.
            function: The function to call when the tool is used.
        """
        self.name = name
        self.description = description
        self.function = function

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool function.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.
        """
        return self.function(*args, **kwargs)


class ToolUsingAgent(BaseAgent):
    """
    An implementation of BaseAgent that can use tools to enhance responses.
    This agent can call external tools/APIs to provide more helpful responses.
    """

    def initialize(self) -> None:
        """
        Initialize the Azure OpenAI client and tools.
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
            self.config[
                "system_message"
            ] = "You are a helpful assistant with access to tools."

        self.state["tools"] = self.config.get("tools", [])

        self.tool_pattern = re.compile(r"TOOL\[(.*?)\]\((.*?)\)")

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent.

        Args:
            tool: The tool to add.
        """
        self.state["tools"].append(tool)

    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            name: The name of the tool to get.

        Returns:
            The tool if found, None otherwise.
        """
        for tool in self.state["tools"]:
            if tool.name == name:
                return tool
        return None

    def format_tools_for_prompt(self) -> str:
        """
        Format tools as a string for the prompt.

        Returns:
            Formatted tools string.
        """
        if not self.state["tools"]:
            return ""

        tools_str = "You have access to the following tools:\n\n"

        for tool in self.state["tools"]:
            tools_str += f"- {tool.name}: {tool.description}\n"

        tools_str += "\nTo use a tool, use the following format: TOOL[tool_name](tool_arguments)\n"
        tools_str += "For example: TOOL[calculator](2+2)\n"
        tools_str += "You can use multiple tools in your response.\n"

        return tools_str

    def process_tool_calls(self, text: str) -> str:
        """
        Process tool calls in the text.

        Args:
            text: The text to process.

        Returns:
            The processed text with tool calls replaced by their results.
        """

        def replace_tool_call(match):
            tool_name = match.group(1)
            tool_args = match.group(2)

            tool = self.get_tool_by_name(tool_name)
            if not tool:
                return f"[Error: Tool '{tool_name}' not found]"

            try:
                result = tool.execute(tool_args)
                return f"[{tool_name} result: {result}]"
            except Exception as e:
                return f"[Error executing tool '{tool_name}': {str(e)}]"

        return self.tool_pattern.sub(replace_tool_call, text)

    def process_message(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Process a user message and generate a streaming response with tool usage.

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

        tools_str = self.format_tools_for_prompt()

        messages = []

        system_message = self.config.get(
            "system_message", "You are a helpful assistant."
        )
        if tools_str:
            system_message = f"{system_message}\n\n{tools_str}"

        messages.append({"role": "system", "content": system_message})

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
            buffer = ""

            for chunk in stream:
                if (
                    chunk.choices
                    and len(chunk.choices) > 0
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content is not None
                ):
                    content = chunk.choices[0].delta.content
                    full_response += content
                    buffer += content

                    if "TOOL[" in buffer and "](" in buffer and ")" in buffer:
                        processed_buffer = self.process_tool_calls(buffer)
                        if processed_buffer != buffer:
                            yield processed_buffer
                            buffer = ""
                        else:
                            yield content
                            buffer = ""
                    else:
                        yield content

            if buffer:
                processed_buffer = self.process_tool_calls(buffer)
                if processed_buffer != buffer:
                    yield processed_buffer

            processed_response = self.process_tool_calls(full_response)

            return {
                "response": processed_response,
                "model": self.config["deployment_name"],
                "timestamp": time.time(),
                "tools_used": bool(self.tool_pattern.search(full_response)),
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
        Process a user message and return a complete response (non-streaming) with tool usage.

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

        tools_str = self.format_tools_for_prompt()

        messages = []

        system_message = self.config.get(
            "system_message", "You are a helpful assistant."
        )
        if tools_str:
            system_message = f"{system_message}\n\n{tools_str}"

        messages.append({"role": "system", "content": system_message})

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

            processed_response = self.process_tool_calls(content)

            return {
                "response": processed_response,
                "model": self.config["deployment_name"],
                "timestamp": time.time(),
                "finish_reason": response.choices[0].finish_reason,
                "tools_used": bool(self.tool_pattern.search(content)),
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
            "tool_usage",
            "external_api_integration",
        ]
