from langchain_core.language_models import BaseLanguageModel
from logger import setup_logger

logger = setup_logger(__name__)

class LLMClient:
    def __init__(self, model: BaseLanguageModel):
        self.model = model

    def query(self, prompt):
        try:
            response = self.model.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise
        

