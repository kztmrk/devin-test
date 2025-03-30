import os

import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv():
        pass
    print("Warning: python-dotenv not found. Environment variables will not be loaded from .env file.")

from openai import AzureOpenAI

st.set_page_config(
    page_title="Azure OpenAI ストリーミングチャットボット",
    page_icon="💬",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-35-turbo"  # Azure OpenAIのデプロイメント名

if "azure_api_key" not in st.session_state:
    st.session_state.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

if "azure_endpoint" not in st.session_state:
    st.session_state.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")

if "azure_api_version" not in st.session_state:
    st.session_state.azure_api_version = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2023-05-15"
    )

if "azure_deployment" not in st.session_state:
    st.session_state.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

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
        st.error(
            "Azure OpenAIの設定が不完全です。サイドバーで必要な情報を入力してください。"
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            client = AzureOpenAI(
                api_key=st.session_state.azure_api_key,
                api_version=st.session_state.azure_api_version,
                azure_endpoint=st.session_state.azure_endpoint,
            )

            stream = client.chat.completions.create(
                model=st.session_state.azure_deployment,  # デプロイメント名を指定
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            full_response = (
                "申し訳ありません、エラーが発生しました。もう一度お試しください。"
            )
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

st.markdown("---")
st.markdown(
    "このアプリケーションはStreamlitとAzure OpenAI APIを使用しています。APIキーは安全に管理してください。"
)
