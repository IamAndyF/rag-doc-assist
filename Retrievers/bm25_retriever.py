
from langchain.retrievers import BM25Retriever

def build_bm25_retriever(docs, k):
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever

    