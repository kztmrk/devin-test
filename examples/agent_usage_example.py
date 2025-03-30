import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.streamlit_app.agent.factory import AgentFactory
from dotenv import load_dotenv

load_dotenv()

def main():
    config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    }
    
    agent = AgentFactory.create_agent("azure_openai", config)
    
    print("Azure OpenAI Agent 使用例")
    print("------------------------")
    print("メッセージを入力してください（終了するには 'exit' と入力）：")
    
    context = []
    
    while True:
        message = input("> ")
        if message.lower() == "exit":
            break
        
        context.append({"role": "user", "content": message})
        
        print("応答：")
        
        full_response = ""
        for chunk in agent.process_message(message, context):
            full_response += chunk
            print(chunk, end="", flush=True)
        print("\n")
        
        context.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
