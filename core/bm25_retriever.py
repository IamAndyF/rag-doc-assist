# BM25 retriever
from langchain.retrievers import BM25Retriever

class myBM25Retriever:
    def __init__(self, docs, k=5):
        self.retriever = BM25Retriever.from_documents(docs)
        self.retriever.k = k
    
    def get_bm25_retriever(self):
        return self.retriever
