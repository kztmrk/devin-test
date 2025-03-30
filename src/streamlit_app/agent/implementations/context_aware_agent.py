import time
from typing import Any, Dict, Generator, List, Optional

from openai import AzureOpenAI

from ..base.base_agent import BaseAgent


class ContextAwareAgent(BaseAgent):
    """
    An implementation of BaseAgent that enhances responses with document context.
    This agent uses document context to provide more informed and relevant responses.
    """

    def initialize(self) -> None:
        """
        Initialize the Azure OpenAI client and context management.
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
            ] = "You are a helpful assistant with access to relevant context."

        self.state["documents"] = self.config.get("documents", [])
        self.state["context_window_size"] = self.config.get("context_window_size", 4000)

    def add_document(self, document: Dict[str, Any]) -> None:
        """
        Add a document to the agent's context.

        Args:
            document: A dictionary containing document information.
                     Must include 'content' and 'metadata' keys.
        """
        if "content" not in document or "metadata" not in document:
            raise ValueError("Document must include 'content' and 'metadata' keys.")

        self.state["documents"].append(document)

    def retrieve_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the query.

        Args:
            query: The query to retrieve relevant documents for.

        Returns:
            A list of relevant documents.
        """
        relevant_docs = []

        for doc in self.state["documents"]:
            if any(
                keyword in doc["content"].lower() for keyword in query.lower().split()
            ):
                relevant_docs.append(doc)

        relevant_docs.sort(
            key=lambda doc: sum(
                1
                for keyword in query.lower().split()
                if keyword in doc["content"].lower()
            ),
            reverse=True,
        )

        return relevant_docs[:3]  # Return top 3 most relevant documents

    def format_context_for_prompt(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Format relevant documents as context for the prompt.

        Args:
            relevant_docs: A list of relevant documents.

        Returns:
            Formatted context string.
        """
        if not relevant_docs:
            return ""

        context_str = (
            "Here is some relevant information that might help you answer:\n\n"
        )

        for i, doc in enumerate(relevant_docs):
            context_str += f"Document {i + 1}: \n"
            context_str += f"Title: {doc['metadata'].get('title', 'Untitled')}\n"
            context_str += f"Content: {doc['content'][:500]}...\n\n"

        return context_str

    def process_message(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Process a user message and generate a streaming response with document context.

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

        relevant_docs = self.retrieve_relevant_context(message)

        context_str = self.format_context_for_prompt(relevant_docs)

        messages = []

        if "system_message" in self.config:
            messages.append(
                {"role": "system", "content": self.config["system_message"]}
            )

        if context:
            messages.extend(context)

        enhanced_message = f"{message}\n\n{context_str}" if context_str else message
        messages.append({"role": "user", "content": enhanced_message})

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
                "context_used": bool(relevant_docs),
                "num_context_docs": len(relevant_docs),
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
        Process a user message and return a complete response (non-streaming) with document context.

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

        relevant_docs = self.retrieve_relevant_context(message)

        context_str = self.format_context_for_prompt(relevant_docs)

        messages = []

        if "system_message" in self.config:
            messages.append(
                {"role": "system", "content": self.config["system_message"]}
            )

        if context:
            messages.extend(context)

        enhanced_message = f"{message}\n\n{context_str}" if context_str else message
        messages.append({"role": "user", "content": enhanced_message})

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
                "context_used": bool(relevant_docs),
                "num_context_docs": len(relevant_docs),
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
            "document_context",
            "context_retrieval",
        ]
