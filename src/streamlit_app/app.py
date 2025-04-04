import os
import sys

import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.streamlit_app.agent.manager import AgentManager  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:

    def load_dotenv():
        pass

    print(
        "Warning: python-dotenv not found. Environment variables will not be loaded from .env file."
    )

st.set_page_config(
    page_title="Azure OpenAI ストリーミングチャットボット",
    page_icon="💬",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False  # デフォルトはオフ

if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-35-turbo"  # Azure OpenAIのデプロイメント名

if "azure_api_key" not in st.session_state:
    st.session_state.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

if "azure_endpoint" not in st.session_state:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if endpoint and not endpoint.endswith("/"):
        endpoint += "/"
    st.session_state.azure_endpoint = endpoint

if "azure_api_version" not in st.session_state:
    st.session_state.azure_api_version = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2023-05-15"
    )

if "azure_deployment" not in st.session_state:
    st.session_state.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = AgentManager()

if "agent_type" not in st.session_state:
    st.session_state.agent_type = "azure_openai"

with st.sidebar:
    st.title("設定")

    api_key = st.text_input(
        "Azure OpenAI APIキー",
        value=st.session_state.azure_api_key,
        type="password",
        help="Azure OpenAI APIキーを入力してください。環境変数で設定されている場合は自動的に読み込まれます。",
    )
    if api_key:
        st.session_state.azure_api_key = api_key

    endpoint = st.text_input(
        "Azure OpenAIエンドポイント",
        value=st.session_state.azure_endpoint,
        help="Azure OpenAIエンドポイントを入力してください。例: https://your-resource-name.openai.azure.com/",
    )
    if endpoint:
        if not endpoint.endswith("/"):
            endpoint = endpoint + "/"

        if not endpoint.startswith("https://") or ".openai.azure.com/" not in endpoint:
            st.warning(
                "エンドポイントは通常 'https://your-resource-name.openai.azure.com/' の形式です。"
            )

        st.session_state.azure_endpoint = endpoint

    deployment = st.text_input(
        "デプロイメント名",
        value=st.session_state.azure_deployment,
        help="Azure OpenAIのデプロイメント名を入力してください。",
    )
    if deployment:
        st.session_state.azure_deployment = deployment

    api_version = st.text_input(
        "APIバージョン",
        value=st.session_state.azure_api_version,
        help="Azure OpenAI APIのバージョンを入力してください。",
    )
    if api_version:
        st.session_state.azure_api_version = api_version

    if st.button("会話をリセット"):
        st.session_state.messages = []
        st.success("会話履歴をリセットしました。")

    with st.expander("開発者オプション"):
        debug_mode = st.checkbox("デバッグモード", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            st.rerun()

        available_agents = st.session_state.agent_manager.get_available_agents()
        agent_options = list(available_agents.keys())
        agent_descriptions = list(available_agents.values())

        agent_type = st.selectbox(
            "エージェントタイプ",
            options=agent_options,
            format_func=lambda x: f"{x} - {available_agents[x]}",
            index=agent_options.index(st.session_state.agent_type)
            if st.session_state.agent_type in agent_options
            else 0,
        )

        if agent_type != st.session_state.agent_type:
            st.session_state.agent_type = agent_type
            st.rerun()

        if st.session_state.agent_type == "duckduckgo_search":
            st.subheader("検索設定")

            if "search_enabled" not in st.session_state:
                st.session_state.search_enabled = True

            if "max_search_results" not in st.session_state:
                st.session_state.max_search_results = 3

            if "search_region" not in st.session_state:
                st.session_state.search_region = "jp-ja"

            search_enabled = st.checkbox(
                "検索機能を有効にする", value=st.session_state.search_enabled
            )
            if search_enabled != st.session_state.search_enabled:
                st.session_state.search_enabled = search_enabled

            max_results = st.slider(
                "最大検索結果数",
                min_value=1,
                max_value=10,
                value=st.session_state.max_search_results,
            )
            if max_results != st.session_state.max_search_results:
                st.session_state.max_search_results = max_results

            news_search = st.checkbox(
                "ニュース検索を有効にする", value=st.session_state.get("news_search", True)
            )
            if news_search != st.session_state.get("news_search", True):
                st.session_state.news_search = news_search

            max_refinements = st.slider(
                "検索クエリ最適化回数",
                min_value=0,
                max_value=2,
                value=st.session_state.get("max_query_refinements", 1),
            )
            if max_refinements != st.session_state.get("max_query_refinements", 1):
                st.session_state.max_query_refinements = max_refinements

            use_structured_output = st.checkbox(
                "構造化出力を使用する（一部のモデルでは非対応）",
                value=st.session_state.get("use_structured_output", True),
            )
            if use_structured_output != st.session_state.get(
                "use_structured_output", True
            ):
                st.session_state.use_structured_output = use_structured_output

st.title("Azure OpenAI ストリーミングチャットボット")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("メッセージを入力してください"):
    if (
        not st.session_state.azure_api_key
        or not st.session_state.azure_endpoint
        or not st.session_state.azure_deployment
    ):
        st.error("Azure OpenAIの設定が不完全です。サイドバーで必要な情報を入力してください。")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            if st.session_state.debug_mode:
                st.write("デバッグ情報：")
                st.write(
                    f"API Key: {'設定済み' if st.session_state.azure_api_key else '未設定'}"
                )
                st.write(f"Endpoint: {st.session_state.azure_endpoint}")
                st.write(f"API Version: {st.session_state.azure_api_version}")
                st.write(f"Deployment: {st.session_state.azure_deployment}")
                st.write(f"Agent Type: {st.session_state.agent_type}")
                st.write("注意: API Version 2024-10-21以降では応答形式が変更されている場合があります")

            agent_config = {
                "api_key": st.session_state.azure_api_key,
                "api_version": st.session_state.azure_api_version,
                "azure_endpoint": st.session_state.azure_endpoint,
                "deployment_name": st.session_state.azure_deployment,
            }

            if st.session_state.agent_type == "duckduckgo_search":
                agent_config.update(
                    {
                        "search_enabled": st.session_state.get("search_enabled", True),
                        "max_search_results": st.session_state.get(
                            "max_search_results", 3
                        ),
                        "search_region": st.session_state.get("search_region", "jp-ja"),
                        "news_search": st.session_state.get("news_search", True),
                        "max_query_refinements": st.session_state.get(
                            "max_query_refinements", 1
                        ),
                        "use_structured_output": st.session_state.get(
                            "use_structured_output", True
                        ),
                    }
                )

            try:
                st.session_state.agent_manager.initialize_agent(
                    st.session_state.agent_type, agent_config
                )
            except Exception as e:
                st.error(f"エージェントの初期化に失敗しました: {str(e)}")
                st.stop()

            stream = st.session_state.agent_manager.process_message(
                prompt,
                [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            )

            spinner_placeholder = st.empty()
            is_searching = False
            display_response = ""

            for chunk in stream:
                if "<search_start>" in chunk:
                    is_searching = True
                    search_message = chunk.split("<search_start>")[1].split(
                        "</search_start>"
                    )[0]
                    with spinner_placeholder.container():
                        st.spinner(search_message)
                    chunk = ""

                elif "<search_end>" in chunk:
                    is_searching = False
                    spinner_placeholder.empty()
                    chunk = chunk.replace("<search_end>", "")

                full_response += chunk

                if not is_searching:
                    display_response = full_response
                    message_placeholder.markdown(display_response + "▌")

            spinner_placeholder.empty()
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_msg = str(e)
            st.error(f"エラーが発生しました: {error_msg}")

            if "authentication" in error_msg.lower() or "401" in error_msg:
                st.warning("認証エラーが発生しました。以下を確認してください：")
                st.warning("1. APIキーが正しいことを確認してください")
                st.warning(
                    "2. エンドポイントURLが正しいことを確認してください（https://your-resource-name.openai.azure.com/）"
                )
                st.warning("3. デプロイメント名が正しいことを確認してください")
            elif "not found" in error_msg.lower() or "404" in error_msg:
                st.warning("リソースが見つかりませんでした。デプロイメント名が正しいことを確認してください")
            elif "index" in error_msg.lower() and "range" in error_msg.lower():
                st.warning("API応答の処理中にエラーが発生しました。APIバージョンに互換性の問題がある可能性があります。")
                st.warning(f"現在のAPIバージョン: {st.session_state.azure_api_version}")
                st.warning("推奨APIバージョン: 2023-05-15")
            elif "agent" in error_msg.lower():
                st.warning("エージェント関連のエラーが発生しました。以下を確認してください：")
                st.warning(
                    f"1. 選択したエージェントタイプ ({st.session_state.agent_type}) が正しいことを確認してください"
                )
                st.warning("2. エージェントの設定が正しいことを確認してください")

            full_response = "申し訳ありません、エラーが発生しました。もう一度お試しください。"
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

st.markdown("---")
st.markdown("このアプリケーションはStreamlitとAzure OpenAI APIを使用しています。APIキーは安全に管理してください。")
