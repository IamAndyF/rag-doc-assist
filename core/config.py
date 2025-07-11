import os
from dotenv import load_dotenv

def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    env_path = os.path.join(parent_dir, '.env')
    load_dotenv(env_path)
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        print(f"✅ Loaded OPENAI_API_KEY")
    else:
        print("❌ OPENAI_API_KEY not found in environment.")

    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "persist_dir": "./chroma_store"
    }
