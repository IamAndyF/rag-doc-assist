from collections import namedtuple

import tiktoken
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
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


class RAGEngine:
    def __init__(
        self,
        api_key,
        persist_dir,
        folder_path,
        chunk_size,
        chunk_overlap,
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
    ):
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.folder_path = folder_path
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = OpenAI(temperature=0, api_key=self.api_key, model_name=self.model)
        self.llm_client = LLMClient(self.llm)

    def format_docs_token_limited(
        self, docs, model_name="gpt-4o-mini", max_context_tokens=1000
    ):
        encoding = tiktoken.encoding_for_model(model_name)
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
        embeddings = OpenAIEmbeddings(model=self.embedding_model, api_key=self.api_key)
        vectordb = Chroma(
            persist_directory=self.persist_dir, embedding_function=embeddings
        )

        loader_agent = LoaderAgent(self.llm_client)

        ingestor = DocumentIngestion(self.folder_path, loader_agent, vectordb)

        # Load existing documents and filter new ones
        new_docs = ingestor.ingest_new_documents()

        if new_docs:
            chunks = TokenTextSplitter(
                self.chunk_size, self.chunk_overlap
            ).split_documents(new_docs)
            vectordb.add_documents(chunks)
            logger.info(f"Added {len(chunks)} new chunks to vector store")
        else:
            logger.info("No new documents added")

        # Build retreiver
        retriever = ContextualCompressionRetriever(
            base_retriever=vectordb.as_retriever(),
            base_compressor=LLMChainExtractor.from_llm(self.llm),
        )

        # Create template
        prompt = PromptTemplate.from_template(
            "Answer the question based only on the context below:\n\n{context}\n\nQuestion: {question}"
            "Be concise with the answer"
        )

        # RAG chain
        rag_chain = RunnableParallel(
            {"question": RunnablePassthrough(), "docs": retriever}
        ) | RunnableLambda(
            lambda x: self.generate_answer_and_sources(x["question"], x["docs"], prompt)
        )

        # With no sources just a direct asnwer use this
        # rag_chain = (
        #     RunnableParallel({
        #         "context": retriever | (lambda docs: self.format_docs_token_limited(docs)),
        #         "question": RunnablePassthrough()
        #     })
        #     | prompt
        #     | self.llm
        #     | StrOutputParser()
        # )

        return rag_chain

    def generate_answer_and_sources(self, question, docs, prompt_template):
        AnswerResult = namedtuple("AnswerResult", ["answer", "sources"])
        context = self.format_docs_token_limited(docs)
        prompt = prompt_template.format(context=context, question=question)
        answer = self.llm.invoke(prompt)
        sources = [doc.metadata.get("filename", "Unknown") for doc in docs]
        return AnswerResult(answer.strip(), list(set(sources)))
