import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Any

load_dotenv()

st.set_page_config(
    page_title="OpenAI ストリーミングチャットボット",
    page_icon="💬",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

with st.sidebar:
    st.title("設定")
    
    api_key = st.text_input("OpenAI APIキー", 
                           value=st.session_state.openai_api_key, 
                           type="password",
                           help="OpenAI APIキーを入力してください。環境変数で設定されている場合は自動的に読み込まれます。")
    
    if api_key:
        st.session_state.openai_api_key = api_key
    
    model = st.selectbox(
        "モデルを選択",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0,
    )
    st.session_state.openai_model = model
    
    if st.button("会話をリセット"):
        st.session_state.messages = []
        st.success("会話履歴をリセットしました。")

st.title("OpenAI ストリーミングチャットボット")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("メッセージを入力してください"):
    if not st.session_state.openai_api_key:
        st.error("OpenAI APIキーが設定されていません。サイドバーでAPIキーを入力してください。")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            client = OpenAI(api_key=st.session_state.openai_api_key)
            
            stream = client.chat.completions.create(
                model=st.session_state.openai_model,
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
            full_response = "申し訳ありません、エラーが発生しました。もう一度お試しください。"
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown("---")
st.markdown("このアプリケーションはStreamlitとOpenAI APIを使用しています。APIキーは安全に管理してください。")
