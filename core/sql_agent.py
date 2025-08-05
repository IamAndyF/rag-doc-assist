from langchain.chains import create_sql_query_chain
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import OpenAI
from core.database_connector import DBConnector
from core.llm_client import LLMClient
from core.config import DATABASE_URL

class SQLAgent:
    def __init__(self, llm_client: LLMClient, db_url = DATABASE_URL):
        self.llm_client = llm_client
        self.db_url = db_url
        self.db_coonector = DBConnector()

    
    def run_query(self, question):
        with self.db_coonector.get_connection(self.db_url) as db:
            schema = self.db_coonector.get_schema(db)
            schema_str = self.db_coonector.format_schema_to_string(schema)

            prompt = f"""You are a smart SQL assistant. You are given a database schema:
            {schema_str}

            Answer the following question about the schema: 
            {question}
            """
            return self.llm_client.invoke(prompt)
            
