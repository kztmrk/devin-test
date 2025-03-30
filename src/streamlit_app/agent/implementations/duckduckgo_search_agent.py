import json
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


class SourceCitation(BaseModel):
    """情報ソースの引用情報を管理するためのモデル"""

    source_id: int = Field(description="引用ソースの識別番号（例：[1], [2]）")
    title: str = Field(description="引用元のタイトル")
    url: str = Field(description="引用元のURL")
    date: Optional[str] = Field(description="公開日時（わかれば）", default=None)
    source_type: str = Field(description="情報ソースの種類（一次情報/二次情報/不明）", default="不明")
    excerpt: str = Field(description="引用する情報の抜粋", default="")


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
また、情報源を引用する際は[1]、[2]のように番号を付けて、どの情報がどの情報源から得られたかを明確にしてください。
回答の最後に「引用文献」セクションを設け、使用した情報源のタイトル、URL、公開日時を列挙してください。
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

        if "use_structured_output" not in self.config:
            self.config["use_structured_output"] = True

        if "citation_format" not in self.config:
            self.config[
                "citation_format"
            ] = "numbered"  # 'numbered', 'footnote', 'inline'のいずれか

        if "include_citations_section" not in self.config:
            self.config["include_citations_section"] = True

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

    def _ask_llm_with_structured_output(
        self, prompt: str, json_schema: dict, temperature: float = 0.0
    ) -> dict:
        """
        Azure OpenAI APIを使用して構造化出力を返すプロンプトに対する回答を取得します。
        構造化出力をサポートしていないモデルの場合は、通常のプロンプトにスキーマ情報を含めて実行します。

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

        if not self.config.get("use_structured_output", True):
            return self._ask_llm_with_structured_output_fallback(
                prompt, json_schema, temperature
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

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            error_str = str(e)
            if (
                "response_format.schema" in error_str
                and "unknown_parameter" in error_str
            ):
                print(
                    f"モデル {self.config['deployment_name']} は構造化出力をサポートしていません。フォールバック方式を使用します。"
                )
                self.config["use_structured_output"] = False
                return self._ask_llm_with_structured_output_fallback(
                    prompt, json_schema, temperature
                )

            print(f"Error calling Azure OpenAI API with structured output: {error_str}")
            return {}

    def _ask_llm_with_structured_output_fallback(
        self, prompt: str, json_schema: dict, temperature: float = 0.0
    ) -> dict:
        """
        構造化出力をサポートしていないモデル用のフォールバック実装。
        スキーマの情報をプロンプトに埋め込み、JSONフォーマットでの回答を要求します。

        Args:
            prompt: 元のプロンプト
            json_schema: JSONスキーマ
            temperature: 生成の多様性

        Returns:
            パースされたJSON（辞書形式）
        """
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        schema_prompt = "以下のJSONスキーマに従って回答してください：\n"
        schema_prompt += "{\n"
        for prop_name, prop_info in properties.items():
            schema_prompt += f'  "{prop_name}": {{\n'
            schema_prompt += f"    \"type\": \"{prop_info.get('type', 'string')}\",\n"
            schema_prompt += (
                f"    \"description\": \"{prop_info.get('description', '')}\"\n"
            )
            schema_prompt += "  },\n"
        schema_prompt += "}\n\n"

        if required:
            schema_prompt += f"必須フィールド: {', '.join(required)}\n\n"

        schema_prompt += "必ず有効なJSONオブジェクトとして回答してください。"

        enhanced_prompt = f"{prompt}\n\n{schema_prompt}"

        response_text = self._ask_llm(enhanced_prompt, temperature)

        try:
            json_str = self._extract_json_from_text(response_text)
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON解析エラー: {e}, 回答: {response_text[:100]}...")
            return {}

    def _extract_json_from_text(self, text: str) -> str:
        """
        テキストからJSON部分を抽出します。

        Args:
            text: 解析するテキスト

        Returns:
            抽出されたJSON文字列
        """
        start_idx = text.find("{")
        if start_idx == -1:
            return "{}"

        open_braces = 0
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                open_braces += 1
            elif text[i] == "}":
                open_braces -= 1
                if open_braces == 0:
                    return text[start_idx : i + 1]

        return "{}"  # 有効なJSONが見つからない場合

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
        - 主要なキーワードも個別にリストアップしてください
        
        ユーザーメッセージ: 「{message}」
        """

        schema = SearchQuery.schema()
        result = self._ask_llm_with_structured_output(prompt, schema)

        if not result or "query" not in result:
            return message.strip()

        query = result.get("query", "").strip()

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
        
        より多くの関連情報を得るために、検索クエリを改善すべきかどうか判断し、改善が必要な場合は新しいクエリを提案してください。
        以下のいずれかの方法を適用できます：
        - より一般的な用語を使用する
        - 別の言い回しを試す
        - 関連キーワードを追加する
        - 制限的すぎる修飾語を削除する
        
        クエリを変更する必要がない場合は、should_refineをfalseにしてください。
        """

        schema = QueryRefinement.schema()
        result = self._ask_llm_with_structured_output(prompt, schema)

        if not result:
            return None

        should_refine = result.get("should_refine", False)
        refined_query = result.get("refined_query", "").strip()

        if should_refine and refined_query and refined_query != original_query:
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
        
        - 日付が見つからない場合は、date_foundをfalseにしてください
        - 複数の日付がある場合は、最も最近のものを選択してください
        - 日付のフォーマットを指定してください（例：YYYY年MM月DD日、YYYY/MM/DD など）
        """

        schema = DateExtraction.schema()
        result = self._ask_llm_with_structured_output(prompt, schema)

        if not result or not result.get("date_found", False):
            return None

        return result.get("date", None)

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
        分類の確信度（0.0-1.0）と理由も説明してください。
        """

        schema = SourceClassification.schema()
        result = self._ask_llm_with_structured_output(prompt, schema)

        if not result:
            return "不明"

        source_type = result.get("source_type", "不明")

        return source_type

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

        citations = self.generate_citations(results)
        formatted += "\n引用形式:\n"
        for citation in citations:
            formatted += f"[{citation.source_id}] {citation.title}. "
            if citation.date:
                formatted += f"({citation.date}). "
            formatted += f"{citation.url}\n"

        return formatted

    def generate_citations(
        self, search_results: List[Dict[str, str]]
    ) -> List[SourceCitation]:
        """
        検索結果から引用情報を生成します。

        Args:
            search_results: 検索結果のリスト

        Returns:
            引用情報のリスト
        """
        citations = []

        for i, result in enumerate(search_results, 1):
            citation = SourceCitation(
                source_id=i,
                title=result.get("title", ""),
                url=result.get("href", ""),
                date=result.get("date", None),
                source_type=result.get("source_type", "不明"),
                excerpt=result.get("body", "")[:150] + "..."
                if len(result.get("body", "")) > 150
                else result.get("body", ""),
            )
            citations.append(citation)

        return citations

    def format_citation_instructions(self, citations: List[SourceCitation]) -> str:
        """
        引用情報から引用指示文を生成します。

        Args:
            citations: 引用情報のリスト

        Returns:
            引用指示文
        """
        citation_instruction = """
検索結果の情報を引用する場合は、[1]、[2]のようにソース番号を文中に含めてください。
回答の最後に「引用文献」セクションを設け、以下の形式で情報源を列挙してください:

引用文献:
"""
        for citation in citations:
            citation_instruction += f"[{citation.source_id}] {citation.title}"
            if citation.date:
                citation_instruction += f" ({citation.date})"
            citation_instruction += f". {citation.url}\n"

        return citation_instruction

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

        if message.lower().startswith(("source:", "ソース:", "出典:", "引用:")):
            source_generator = self.handle_source_command(message)
            response_dict = None

            for chunk in source_generator:
                yield chunk

            try:
                response_dict = source_generator.send(None)
            except StopIteration as e:
                if hasattr(e, "value") and isinstance(e.value, dict):
                    response_dict = e.value
                else:
                    response_dict = {
                        "response": "ソース情報の処理中にエラーが発生しました。",
                        "error": "無効な応答形式",
                        "timestamp": time.time(),
                    }

            if not isinstance(response_dict, dict):
                response_dict = {
                    "response": str(response_dict)
                    if response_dict
                    else "ソース情報の処理中にエラーが発生しました。",
                    "error": "無効な応答形式",
                    "timestamp": time.time(),
                }

            return response_dict

        search_results = []
        search_info = ""

        if self.should_search(message):
            query = self.generate_search_query(message)

            yield f"<search_start>🔍 「{query}」を検索中...</search_start>"
            search_results = self.perform_search(query)

            if (
                len(search_results) < 2
                and self.config.get("max_query_refinements", 1) > 0
            ):
                refined_query = self.refine_search_query(query, search_results)
                if refined_query and refined_query != query:
                    yield f"<search_start>🔍 検索クエリを「{refined_query}」に改善して再検索中...</search_start>"
                    self.state["refined_query"] = refined_query
                    search_results = self.perform_search(refined_query)

            yield "<search_end>"

            if search_results:
                search_info = self.format_search_results(search_results)
                yield f"{search_info}\n\n回答を生成中...\n\n"

        messages = []

        system_message = self.config.get(
            "system_message", "あなたは外部情報検索機能を持つ優秀なアシスタントです。"
        )

        if search_results:
            citations = self.generate_citations(search_results)
            self.state["citations"] = citations

            citation_instruction = self.format_citation_instructions(citations)

            system_message += f"\n\n以下の検索結果を参考にして回答してください。検索結果が質問に関連しない場合は無視してください。\n\n{search_info}\n\n{citation_instruction}"

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
                "structured_output_supported": self.config.get(
                    "use_structured_output", True
                ),
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
            citations = self.generate_citations(search_results)
            self.state["citations"] = citations

            citation_instruction = self.format_citation_instructions(citations)

            system_message += f"\n\n以下の検索結果を参考にして回答してください。検索結果が質問に関連しない場合は無視してください。\n\n{search_info}\n\n{citation_instruction}"

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
                "structured_output_supported": self.config.get(
                    "use_structured_output", True
                ),
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

    def extract_source_content(self, source_id: int) -> Optional[Dict[str, Any]]:
        """
        指定されたソースIDの情報を抽出・展開します。

        Args:
            source_id: 抽出するソースのID

        Returns:
            ソースの詳細情報を含む辞書、見つからない場合はNone
        """
        if not self.state.get("last_search_results"):
            return None

        search_results = self.state.get("last_search_results")

        if not search_results or source_id <= 0 or source_id > len(search_results):
            return None

        result = search_results[source_id - 1]

        extracted_info = {
            "title": result.get("title", ""),
            "full_content": result.get("body", ""),
            "url": result.get("href", ""),
            "date": result.get("date", None),
            "source_type": result.get("source_type", "不明"),
            "source_id": source_id,
        }

        prompt = f"""
        以下の情報源から主要なポイントを抽出し、内容を要約してください：
        
        タイトル: {extracted_info["title"]}
        内容: {extracted_info["full_content"]}
        """

        try:
            key_points_analysis = self._ask_llm(prompt)
            extracted_info["key_points"] = key_points_analysis
        except Exception as e:
            extracted_info["key_points"] = "コンテンツ分析中にエラーが発生しました。"

        return extracted_info

    def handle_source_command(
        self, message: str
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        ソース展開コマンドを処理します。

        Args:
            message: ユーザーメッセージ (例: "source:1" or "ソース:2")

        Returns:
            展開されたソース情報を含むジェネレーター
        """
        try:
            if message.lower().startswith(("source:", "ソース:", "出典:", "引用:")):
                source_id_str = message.split(":", 1)[1].strip()
                try:
                    source_id = int(source_id_str)
                except ValueError:
                    yield f"エラー: 有効なソースIDを指定してください。例: ソース:1"
                    return {
                        "response": "エラー: 有効なソースIDを指定してください。",
                        "error": "無効なソースID",
                        "timestamp": time.time(),
                    }

                source_info = self.extract_source_content(source_id)

                if not source_info:
                    yield f"エラー: ソースID {source_id} が見つかりません。"
                    return {
                        "response": f"エラー: ソースID {source_id} が見つかりません。",
                        "error": "ソースが見つかりません",
                        "timestamp": time.time(),
                    }

                response = f"# ソース {source_id}: {source_info['title']}\n\n"
                response += f"**URL**: {source_info['url']}\n"
                if source_info.get("date"):
                    response += f"**公開日**: {source_info['date']}\n"
                response += f"**情報の種類**: {source_info['source_type']}\n\n"
                response += f"## 内容\n{source_info['full_content']}\n\n"
                response += f"## 主要ポイント\n{source_info['key_points']}\n"

                yield response

                return {
                    "response": response,
                    "source_info": source_info,
                    "timestamp": time.time(),
                }

            return {
                "response": "",
                "is_source_command": False,
                "timestamp": time.time(),
            }

        except Exception as e:
            error_message = f"ソース展開中にエラーが発生しました: {str(e)}"
            yield error_message
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
            "source_citation",
            "source_content_extraction",
        ]
