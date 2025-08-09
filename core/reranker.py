from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class RerankModel:
    def __init__(self, rerank_model, max_length=512, num_labels=1):
        self.reranker_model = CrossEncoder(rerank_model, max_length=max_length, num_labels=num_labels)
        
    def rerank_documents(self, query, docs: list[Document], top_k=5):
        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.reranker_model.predict(pairs)

        ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in ranked_docs[:top_k]]
