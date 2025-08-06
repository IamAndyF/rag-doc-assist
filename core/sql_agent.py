from core.config import DATABASE_URL
from core.database_connector import DBConnector
from core.llm_client import LLMClient
from logger import logger


class SQLAgent:
    def __init__(self, llm_client: LLMClient, db_url=DATABASE_URL):
        self.llm_client = llm_client
        self.db_url = db_url
        self.db_connector = DBConnector()

    def run_query(self, question):
        try:
            with self.db_connector.get_connection(self.db_url) as db:
                schema = self.db_connector.get_schema(db)
                schema_str = self.db_connector.format_schema_to_string(schema)

                prompt = f"""You are a smart SQL assistant. You are given a database schema:
                {schema_str}

                Answer the following question about the schema: 
                {question}
                """

                return self.llm_client.query(prompt)

        except Exception as e:
            logger.error(f"SQL Agent error: {e}")
