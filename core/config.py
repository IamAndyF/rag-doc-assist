import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_store")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./data")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


class DatabaseConfig:
    def __init__(self):
        # Configuration for the database connection
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.dbname = os.getenv("DB_NAME", "tradeall")
        self.user = os.getenv("DB_USER", "")
        self.password = os.getenv("DB_PASSWORD", "")

    @property
    def connection_string(self):
        return f"dbname='{self.dbname}' user='{self.user}' host='{self.host}' port='{self.port}' password='{self.password}'"


DATABASE_URL = DatabaseConfig().connection_string
