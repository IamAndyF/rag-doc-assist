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

                prompt = f"""
                    You are an intelligent and helpful SQL assistant. You are given a PostgreSQL database schema below.

                    Use this schema to answer questions about the data.

                    ---
                    ### Database Schema
                    {schema_str}
                    ---

                    ### Question:
                    {question}

                    ### Instructions:
                    - If the user asks about the schema (e.g., "what tables exist", "describe the users table"), answer in **clear natural language** with appropriate formatting (e.g., markdown tables).
                    - If the user asks a question that requires a SQL query, provide:
                    1. A brief explanation of what you're doing
                    2. The SQL query in a code block
                    3. Optionally, what the result would look like

                    Respond clearly and avoid overly technical jargon unless asked.
                    """

                return self.llm_client.query(prompt)

        except Exception as e:
            logger.error(f"SQL Agent error: {e}")
