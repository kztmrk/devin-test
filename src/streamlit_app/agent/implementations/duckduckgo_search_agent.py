import time
from typing import Any, Dict, Generator, List, Optional

from duckduckgo_search import DDGS
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from ..base.base_agent import BaseAgent


class SearchDecision(BaseModel):
    """検索の必要性を判断するためのモデル"""
    should_search: bool = Field(description="検索が必要かどうかを示すブール値")
    reason: str = Field(description="判断の理由")


class SearchQuery(BaseModel):
    """検索クエリを生成するためのモデル"""
    query: str = Field(description="生成された検索クエリ")
    keywords: list[str] = Field(description="抽出された主要なキーワード", default_factory=list)


class QueryRefinement(BaseModel):
    """検索クエリを最適化するためのモデル"""
    should_refine: bool = Field(description="クエリを最適化する必要があるかどうか")
    refined_query: str = Field(description="最適化されたクエリ（最適化が不要な場合は空文字列）")
    reason: str = Field(description="最適化の理由または不要と判断した理由")


class DateExtraction(BaseModel):
    """日付情報を抽出するためのモデル"""
    date_found: bool = Field(description="日付が見つかったかどうか")
    date: str = Field(description="抽出された日付情報（見つからなかった場合は空文字列）")
    format: str = Field(description="日付のフォーマット（例：YYYY/MM/DD）", default="")


class SourceClassification(BaseModel):
    """情報ソースを分類するためのモデル"""
    source_type: str = Field(description="「一次情報」、「二次情報」、または「不明」")
    confidence: float = Field(description="分類の確信度（0.0-1.0）", ge=0.0, le=1.0)
    reason: str = Field(description="分類の理由")


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

    def _ask_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Azure OpenAI APIを使用して単一のプロンプトに対する回答を取得します。

        Args:
            prompt: 質問やタスクを含むプロンプト
            temperature: 生成の多様性（0.0は決定論的、1.0は創造的）

        Returns:
            LLMからの回答テキスト
        """
        if not self.client:
            raise ValueError(
                "Azure OpenAI client is not initialized. Please check your configuration."
            )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.config["deployment_name"],
                messages=messages,
                temperature=temperature,
                stream=False,
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Azure OpenAI API: {str(e)}")
            return ""
            
    def _ask_llm_with_structured_output(self, prompt: str, json_schema: dict, temperature: float = 0.0) -> dict:
        """
        Azure OpenAI APIを使用して構造化出力を返すプロンプトに対する回答を取得します。
        
        Args:
            prompt: 質問やタスクを含むプロンプト
            json_schema: 返される構造化JSONのスキーマ
            temperature: 生成の多様性（0.0は決定論的、1.0は創造的）
            
        Returns:
            構造化されたJSONレスポンス（辞書形式）
        """
        if not self.client:
            raise ValueError(
                "Azure OpenAI client is not initialized. Please check your configuration."
            )
            
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.config["deployment_name"],
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object", "schema": json_schema},
                stream=False,
            )
            
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error calling Azure OpenAI API with structured output: {str(e)}")
            return {}

    def should_search(self, message: str) -> bool:
        """
        指定されたメッセージに対して検索を行うべきかどうかを判断します。

        Args:
            message: 分析するユーザーメッセージ

        Returns:
            検索を行うべき場合はTrue、そうでない場合はFalse
        """
        if not self.config.get("search_enabled", True):
            return False

        if message.strip().lower().startswith(("検索:", "search:")):
            return True

        prompt = f"""
        あなたはユーザーの質問を分析し、外部情報の検索が必要かどうかを判断するアシスタントです。
        
        以下のような場合、外部検索が必要と判断してください：
        - 最新の情報やニュースを求めている質問
        - 特定の事実や統計に関する質問
        - 特定の場所、人物、イベントなどに関する具体的な情報を求める質問
        - 「調べて」「教えて」などの明示的な情報要求を含む質問
        
        以下のような場合は外部検索が不要です：
        - 一般的な概念や定義に関する質問
        - 個人的な意見や提案を求める質問
        - 会話の継続や挨拶
        
        「{message}」
        
        このユーザーメッセージは外部情報の検索が必要かどうかを判断し、理由も説明してください。
        """

        schema = SearchDecision.schema()
        result = self._ask_llm_with_structured_output(prompt, schema)
        
        if not result:
            return False
            
        return result.get("should_search", False)

    def generate_search_query(self, message: str) -> str:
        """
        ユーザーメッセージから検索クエリを生成します。

        Args:
            message: クエリを生成するためのユーザーメッセージ

        Returns:
            検索クエリ文字列
        """
        if message.strip().lower().startswith(("検索:", "search:")):
            return message.split(":", 1)[1].strip()

        prompt = f"""
        あなたは検索エンジンの専門家で、効果的な検索クエリを作成するスキルを持っています。
        
        以下のユーザーメッセージを分析し、最も関連性の高い情報を見つけるための最適な検索クエリを生成してください。
        
        - 簡潔で具体的なキーワードを使用する
        - 不要な言葉（「について教えて」など）は省略する
        - 重要なキーワードだけを含める
        - 検索エンジンで最良の結果を得られるように最適化する
        - 日本語のクエリを生成する
        
        「{message}」
        
        """

        query = self._ask_llm(prompt).strip()

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
        初期検索結果に基づいて検索クエリを最適化します。

        Args:
            original_query: 元の検索クエリ
            results: 初期検索結果

        Returns:
            最適化が必要な場合は最適化されたクエリ、そうでない場合はNone
        """
        if not results or len(results) >= self.config.get("max_search_results", 3):
            return None  # 十分な結果がある場合は最適化不要

        results_summary = "\n".join(
            [
                f"タイトル: {result.get('title', '')}\n内容: {result.get('body', '')[:100]}..."
                for result in results
            ]
        )

        prompt = f"""
        あなたは検索クエリ最適化の専門家です。検索結果が少ない場合に、より良い結果を得るためにクエリを改善します。
        
        元の検索クエリ: 「{original_query}」
        検索結果数: {len(results)}件（最大{self.config.get("max_search_results", 3)}件中）
        
        {results_summary}
        
        より多くの関連情報を得るために、検索クエリを改善してください。以下のいずれかの方法を適用できます：
        - より一般的な用語を使用する
        - 別の言い回しを試す
        - 関連キーワードを追加する
        - 制限的すぎる修飾語を削除する
        
        できるだけ簡潔なクエリを1つだけ提案してください。クエリを変更する必要がない場合は「変更不要」と回答してください。
        
        """

        refined_query = self._ask_llm(prompt).strip()

        if (
            refined_query
            and "変更不要" not in refined_query
            and refined_query != original_query
        ):
            return refined_query

        return None

    def extract_date_info(self, result: Dict[str, str]) -> Optional[str]:
        """
        検索結果から公開日情報を抽出します。

        Args:
            result: 日付を抽出する検索結果

        Returns:
            日付が見つかった場合はその文字列表現、見つからなかった場合はNone
        """
        if "published" in result:
            return result.get("published", None)

        title = result.get("title", "")
        body = result.get("body", "")

        prompt = f"""
        あなたは与えられたテキストから日付情報を抽出する専門家です。
        
        以下のテキストから公開日または最後の更新日と思われる日付情報を抽出してください。
        
        タイトル: {title}
        本文: {body}
        
        - 日付が見つからない場合は「見つかりません」と回答してください
        - 複数の日付がある場合は、最も最近のものを選択してください
        - 日付のフォーマットはそのまま抽出してください（例：2023年3月15日、2023/03/15など）
        
        """

        date_info = self._ask_llm(prompt).strip()

        if "見つかりません" in date_info or not date_info:
            return None

        return date_info

    def classify_information_source(self, result: Dict[str, str]) -> str:
        """
        検索結果を一次情報または二次情報として分類します。

        Args:
            result: 分類する検索結果

        Returns:
            "一次情報"、"二次情報"、または"不明"としての分類
        """
        href = result.get("href", "")
        title = result.get("title", "")
        body = result.get("body", "")

        prompt = f"""
        あなたは情報ソースの分類専門家です。情報を一次情報（直接の情報源）と二次情報（間接的な情報源）に分類します。
        
        一次情報（プライマリーソース）:
        - 政府や公式機関からの直接の発表
        - 企業や組織の公式ウェブサイトやプレスリリース
        - 直接の証言や一次資料
        - 原著論文や研究報告書
        
        二次情報（セカンダリーソース）:
        - ニュースサイトや報道機関による報道
        - ブログ記事やレビュー
        - 解説記事や分析
        - まとめサイトや情報キュレーション
        
        URL: {href}
        タイトル: {title}
        内容: {body[:300]}...
        
        上記の情報ソースを分析し、「一次情報」、「二次情報」、または「不明」に分類してください。
        理由も簡潔に説明してください。
        
        """

        classification_result = self._ask_llm(prompt)

        if "一次情報" in classification_result:
            return "一次情報"
        elif "二次情報" in classification_result:
            return "二次情報"
        else:
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
