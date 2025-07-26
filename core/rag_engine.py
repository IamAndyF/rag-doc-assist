import os
import tiktoken
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from core.document_ingestor import DocumentIngestion
from core.loader_manager import LoaderManager
from core.loader_agent import LoaderAgent

from logger import setup_logger
logger = setup_logger(__name__)


class RAGEngine:
    def __init__(self, openai_api_key, folder_path, persist_dir="./chroma_store"):
        self.openai_api_key = openai_api_key
        self.folder_path = folder_path
        self.persist_dir = persist_dir
        self.llm = OpenAI(temperature=0, api_key=openai_api_key, model_name='gpt-4o-mini')

    def format_docs_token_limited(self, docs, model_name='gpt-4o-mini', max_context_tokens=1000):
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
        #Initialise embeddings and persistent store
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=self.openai_api_key )
        vectordb = Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)

        loader_agent = LoaderAgent(
            llm = self.llm,
            valid_loaders = LoaderManager.valid_loaders,
            extension_loader_map = LoaderManager.extension_loader_map  
        )

        ingestor = DocumentIngestion(
            self.folder_path,
            loader_agent,
            vectordb
        )

        #Load existing documents and filter new ones
        new_docs = ingestor.ingest_new_documents()

        if new_docs:
            chunks = TokenTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(new_docs)
            vectordb.add_documents(chunks)
            logger.info(f"Added {len(chunks)} new chunks to vector store")
        else:
            logger.info("No new documents added")
        
        #Build retreiver
        retriever = ContextualCompressionRetriever(
            base_retriever=vectordb.as_retriever(),
            base_compressor= LLMChainExtractor.from_llm(self.llm)
        )

        #Create template
        prompt = PromptTemplate.from_template(
            "Answer the question based only on the context below:\n\n{context}\n\nQuestion: {question}"
            "Be concise with the answer"
        )   

        #RAG chain
        rag_chain = (
            RunnableParallel({
                "context": retriever | (lambda docs: self.format_docs_token_limited(docs)),
                "question": RunnablePassthrough()
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain