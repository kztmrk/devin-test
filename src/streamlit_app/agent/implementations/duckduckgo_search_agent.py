import time
from typing import Any, Dict, Generator, List, Optional

from duckduckgo_search import DDGS
from openai import AzureOpenAI

from ..base.base_agent import BaseAgent


class DuckDuckGoSearchAgent(BaseAgent):
    """
    An implementation of BaseAgent that uses DuckDuckGo search to enhance responses.
    This agent can search for external information to provide more accurate and up-to-date responses.
    """

    def initialize(self) -> None:
        """
        Initialize the Azure OpenAI client and DuckDuckGo search.
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
            ] = """あなたは外部情報検索機能を持つ優秀なアシスタントです。ユーザーの質問に対して、必要に応じてDuckDuckGoで検索を行い、最新かつ正確な情報を提供します。

情報の信頼性を評価する際は、以下の点に注意してください：
1. 一次情報（公式発表、原著論文など）は二次情報よりも信頼性が高い傾向があります
2. 情報の新しさ（公開日時）を考慮し、最新の情報を優先してください
3. 複数の情報源が一致する情報はより信頼できる可能性があります

回答では、使用した情報の信頼性について言及し、ユーザーが情報の質を判断できるようにしてください。
"""

        if "search_enabled" not in self.config:
            self.config["search_enabled"] = True

        if "max_search_results" not in self.config:
            self.config["max_search_results"] = 3

        if "search_region" not in self.config:
            self.config["search_region"] = "jp-ja"

        if "news_search" not in self.config:
            self.config["news_search"] = True

        if "max_query_refinements" not in self.config:
            self.config["max_query_refinements"] = 1

        self.ddgs = DDGS()
        self.state["last_search_query"] = None
        self.state["last_search_results"] = None
        self.state["refined_query"] = None

    def should_search(self, message: str) -> bool:
        """
        Determine if a search should be performed for the given message.

        Args:
            message: The user message to analyze.

        Returns:
            True if a search should be performed, False otherwise.
        """
        if not self.config.get("search_enabled", True):
            return False

        if message.strip().lower().startswith(("検索:", "search:")):
            return True

        search_indicators = [
            "最新",
            "最近",
            "ニュース",
            "情報",
            "データ",
            "統計",
            "いつ",
            "どこ",
            "誰が",
            "何が",
            "どのように",
            "なぜ",
            "調べて",
            "教えて",
            "知りたい",
            "わかる？",
            "分かる？",
            "latest",
            "recent",
            "news",
            "information",
            "data",
            "statistics",
            "when",
            "where",
            "who",
            "what",
            "how",
            "why",
            "look up",
            "tell me",
            "find",
            "search",
        ]

        for indicator in search_indicators:
            if indicator in message.lower():
                return True

        return False

    def generate_search_query(self, message: str) -> str:
        """
        Generate a search query from the user message.

        Args:
            message: The user message to generate a query from.

        Returns:
            A search query string.
        """
        if message.strip().lower().startswith(("検索:", "search:")):
            return message.split(":", 1)[1].strip()

        query = message.replace("について教えて", "")
        query = query.replace("を調べて", "")
        query = query.replace("は何ですか", "")
        query = query.replace("とは", "")

        if len(query) > 100:
            query = query[:100]

        return query.strip()

    def perform_search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a DuckDuckGo search with the given query.

        Args:
            query: The search query.

        Returns:
            A list of search results.
        """
        try:
            max_results = self.config.get("max_search_results", 3)
            region = self.config.get("search_region", "jp-ja")
            use_news = self.config.get("news_search", True)

            results = []

            if use_news:
                news_results = list(
                    self.ddgs.news(
                        query,
                        region=region,
                        max_results=max(1, max_results // 2),  # 検索結果の半分をニュースから
                    )
                )
                results.extend(news_results)

            text_results = list(
                self.ddgs.text(
                    query,
                    region=region,
                    max_results=max_results - len(results),  # 残りの枠を通常検索で埋める
                )
            )
            results.extend(text_results)

            formatted_results = []
            for result in results:
                date_info = self.extract_date_info(result)
                source_type = self.classify_information_source(result)

                formatted_results.append(
                    {
                        "title": result.get("title", ""),
                        "body": result.get("body", ""),
                        "href": result.get("href", ""),
                        "date": date_info,
                        "source_type": source_type,
                    }
                )

            self.state["last_search_query"] = query
            self.state["last_search_results"] = formatted_results

            return formatted_results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def refine_search_query(
        self, original_query: str, results: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Refine a search query based on initial search results.

        Args:
            original_query: The original search query.
            results: The initial search results.

        Returns:
            A refined query if refinement is needed, None otherwise.
        """
        if not results or len(results) >= self.config.get("max_search_results", 3):
            return None  # 十分な結果がある場合は洗練不要

        if len(results) <= 1:
            generalized_query = original_query

            import re

            generalized_query = re.sub(r"\b\d{4}年?\b", "", generalized_query)
            generalized_query = re.sub(r"\b\d{1,2}月\b", "", generalized_query)

            for term in ["最新の", "最近の", "詳細な", "具体的な"]:
                generalized_query = generalized_query.replace(term, "")

            generalized_query = generalized_query.strip()

            if generalized_query != original_query:
                return generalized_query

        if len(original_query.split()) > 3:
            keywords = " ".join(original_query.split()[:3])
            return keywords

        return None

    def extract_date_info(self, result: Dict[str, str]) -> Optional[str]:
        """
        Extract publication date information from a search result.

        Args:
            result: The search result to extract date from.

        Returns:
            A string representation of the date if found, None otherwise.
        """
        if "published" in result:
            return result.get("published", None)

        body = result.get("body", "")
        date_patterns = [
            r"(\d{4}年\d{1,2}月\d{1,2}日)",  # 2023年3月15日
            r"(\d{4}/\d{1,2}/\d{1,2})",  # 2023/03/15
            r"(\d{4}-\d{1,2}-\d{1,2})",  # 2023-03-15
        ]

        for pattern in date_patterns:
            import re

            match = re.search(pattern, body)
            if match:
                return match.group(1)

        return None

    def classify_information_source(self, result: Dict[str, str]) -> str:
        """
        Classify a search result as primary or secondary information.

        Args:
            result: The search result to classify.

        Returns:
            Classification as "一次情報", "二次情報", or "不明".
        """
        href = result.get("href", "").lower()
        title = result.get("title", "").lower()
        body = result.get("body", "").lower()

        primary_indicators = [
            ".gov.",
            ".go.jp",
            "official",
            "公式",
            "オフィシャル",
            "press release",
            "プレスリリース",
            "発表",
        ]

        secondary_indicators = [
            "news",
            "blog",
            "review",
            "opinion",
            "analysis",
            "ニュース",
            "ブログ",
            "レビュー",
            "まとめ",
            "解説",
            "分析",
        ]

        for indicator in primary_indicators:
            if indicator in href or indicator in title or indicator in body:
                return "一次情報"

        for indicator in secondary_indicators:
            if indicator in href or indicator in title or indicator in body:
                return "二次情報"

        return "不明"

    def format_search_results(self, results: List[Dict[str, str]]) -> str:
        """
        Format search results as a string.

        Args:
            results: The search results to format.

        Returns:
            Formatted search results string.
        """
        if not results:
            return "検索結果はありませんでした。"

        formatted = "検索結果:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   {result['body']}\n"

            if result.get("date"):
                formatted += f"   📅 公開日: {result['date']}\n"

            source_type = result.get("source_type", "不明")
            formatted += f"   📊 情報の種類: {source_type}\n"

            formatted += f"   🔗 出典: {result['href']}\n\n"

        return formatted

    def process_message(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Process a user message and generate a streaming response with search integration.

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

        search_results = []
        search_info = ""

        if self.should_search(message):
            query = self.generate_search_query(message)

            yield f"🔍 「{query}」を検索中...\n\n"
            search_results = self.perform_search(query)

            if (
                len(search_results) < 2
                and self.config.get("max_query_refinements", 1) > 0
            ):
                refined_query = self.refine_search_query(query, search_results)
                if refined_query and refined_query != query:
                    yield f"🔍 検索クエリを「{refined_query}」に改善して再検索中...\n\n"
                    self.state["refined_query"] = refined_query
                    search_results = self.perform_search(refined_query)

            if search_results:
                search_info = self.format_search_results(search_results)
                yield f"{search_info}\n\n回答を生成中...\n\n"

        messages = []

        system_message = self.config.get(
            "system_message", "あなたは外部情報検索機能を持つ優秀なアシスタントです。"
        )

        if search_results:
            system_message += (
                f"\n\n以下の検索結果を参考にして回答してください。検索結果が質問に関連しない場合は無視してください。\n\n{search_info}"
            )

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
                "search_performed": bool(search_results),
                "search_query": self.state.get("last_search_query"),
                "search_results": self.state.get("last_search_results"),
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
        Process a user message and return a complete response (non-streaming) with search integration.

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

        search_results = []
        search_info = ""

        if self.should_search(message):
            query = self.generate_search_query(message)

            search_results = self.perform_search(query)

            if search_results:
                search_info = self.format_search_results(search_results)

        messages = []

        system_message = self.config.get(
            "system_message", "あなたは外部情報検索機能を持つ優秀なアシスタントです。"
        )

        if search_results:
            system_message += (
                f"\n\n以下の検索結果を参考にして回答してください。検索結果が質問に関連しない場合は無視してください。\n\n{search_info}"
            )

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

            return {
                "response": content,
                "model": self.config["deployment_name"],
                "timestamp": time.time(),
                "finish_reason": response.choices[0].finish_reason,
                "search_performed": bool(search_results),
                "search_query": self.state.get("last_search_query"),
                "search_results": self.state.get("last_search_results"),
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
            "external_search",
            "real_time_information",
        ]
