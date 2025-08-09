# Retriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class VectorRetriever:
    def __init__(self, embedding_model, api_key, persist_dir):
        self.embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
        self.vectorstore = Chroma(
            persist_directory=persist_dir, embedding_function=self.embeddings
        )

    def get_vector_retriever(self):
        return self.vectorstore.asretriever()

    def get_embedddings(self):
        return self.embeddings

    def get_vectorstore(self):
        return self.vectorstore
