from collections import namedtuple

import tiktoken
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import OpenAI

from core.document_ingestor import DocumentIngestion
from core.llm_client import LLMClient
from core.loader_agent import LoaderAgent
from logger import logger
from Retrievers.bm25_retriever import build_bm25_retriever
from Retrievers.ensemble_retriever import get_ensemble_retriever
from Retrievers.reranker import RerankModel
from Retrievers.vector_retriever import VectorRetriever
from Retrievers.retriever_cache import load_retriever


class RAGEngine:
    def __init__(
        self,
        api_key,
        persist_dir,
        folder_path,
        chunk_size,
        chunk_overlap,
        cache_path,
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.folder_path = folder_path
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_path = cache_path
        self.llm = OpenAI(temperature=0, api_key=self.api_key, model_name=self.model)
        self.llm_client = LLMClient(self.llm)
        self.vector_retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            api_key=self.api_key,
            persist_dir=self.persist_dir,
        )

    def format_docs_token_limited(
        self, docs, model_name="gpt-4o-mini", max_context_tokens=1000
    ):
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        selected_texts = []
        total_tokens = 0

        for doc in docs:
            text = doc.page_content
            tokens = len(encoding.encode(text))

            if total_tokens + tokens > max_context_tokens:
                break
            selected_texts.append(text)
            total_tokens += tokens

        return "\n\n".join(selected_texts)

    def build_rag_chain(self):
        # Initialise embeddings and persistent store
        vectordb = self.vector_retriever.get_vectorstore()

        loader_agent = LoaderAgent(self.llm_client)

        ingestor = DocumentIngestion(self.folder_path, loader_agent, vectordb)

        # Load existing documents and filter new ones
        new_docs = ingestor.ingest_new_documents()

        if new_docs:
            chunks = TokenTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            ).split_documents(new_docs)
            vectordb.add_documents(chunks)
            logger.info(f"Added {len(chunks)} new chunks to vector store")
        else:
            logger.info("No new documents added")

        # Build retreiver

        all_docs = self._get_all_documents_for_bm25()

        if all_docs:
            sparse_retriever = load_retriever(self.cache_path, build_bm25_retriever, all_docs, k=10)
            vector_retriever = vectordb.as_retriever()

            ensemble_retriever = get_ensemble_retriever(
                dense_retriever=vector_retriever, sparse_retriever=sparse_retriever
            )
        else:
            logger.warning("No documents available for BM25 retriever")
            ensemble_retriever = vectordb.as_retriever()

        reranker = RerankModel(self.reranker_model)

        # Create template
        prompt = PromptTemplate.from_template(
            """Answer the question based ONLY on the context below.
            If the answer is not contained in the context, respond with "I don't know."

            At the end of your answer, list ONLY the filenames from the context that you directly referenced or cited to support your answer.
            Do NOT list files that were not used or referenced.
            List each filename only once.

            Context:
            {context}

            Question:
            {question}

            Format your response like this:
            Answer: <answer text>
            
            Sources:
            - filename1
            - filename2
            """
        )

        # RAG chain
        rag_chain = (
            RunnableParallel(
                {"question": RunnablePassthrough(), "docs": ensemble_retriever}
            )
            | RunnableLambda(
                lambda x: {
                    "question": x["question"],
                    "docs": reranker.rerank_documents(x["question"], x["docs"]),
                }
            )
            | RunnableLambda(
                lambda x: self.generate_answer(
                    x["question"], x["docs"], prompt
                )
            )
        )

        return rag_chain

    def generate_answer(self, question, docs, prompt_template):
        context = self.format_docs_token_limited(docs)
        prompt = prompt_template.format(context=context, question=question)
        answer = self.llm.invoke(prompt).strip()
        return answer

    def _get_all_documents_for_bm25(self):
        """Retrieve all document chunks for BM25 indexing"""
        try:
            vectordb = self.vector_retriever.get_vectorstore()
            results = vectordb.get(include=["documents", "metadatas"])

            if not results["documents"]:
                return []

            from langchain_core.documents import Document

            docs = []
            for doc_text, metadata in zip(results["documents"], results["metadatas"]):
                docs.append(Document(page_content=doc_text, metadata=metadata))
            return docs
        except Exception as e:
            logger.warning(f"Could not retrieve documents for BM25: {e}")
            return []
