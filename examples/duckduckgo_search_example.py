import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.streamlit_app.agent.factory import AgentFactory
from src.streamlit_app.agent.implementations.duckduckgo_search_agent import DuckDuckGoSearchAgent

def main():
    """
    DuckDuckGo検索エージェントの使用例
    """
    config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "search_enabled": True,
        "max_search_results": 3,
        "search_region": "jp-ja",
    }

    agent = AgentFactory.create_agent("duckduckgo_search", config)

    conversation = []

    while True:
        user_input = input("\nあなた: ")
        if user_input.lower() in ["exit", "quit", "終了"]:
            break

        conversation.append({"role": "user", "content": user_input})

        print("\nアシスタント: ", end="")

        full_response = ""
        for chunk in agent.process_message(user_input, conversation):
            print(chunk, end="", flush=True)
            full_response += chunk

        conversation.append({"role": "assistant", "content": full_response})
        print("\n")

if __name__ == "__main__":
    main()
