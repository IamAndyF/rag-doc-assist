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
from core.loader_utils import load_documents_with_metadata, get_existing_hashes, filter_new_hashes

max_context_tokens = 1000

def format_docs_token_limited(docs, model_name='gpt-4o-mini'):
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

def build_rag_engine(openai_api_key, folder_path, persist_dir="./chroma_store"):
    #Initialise embeddings and persistent store
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=openai_api_key 
    )
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Llm loader agent
    llm_for_loader = OpenAI(api_key=openai_api_key, model_name='gpt-4o-mini', temperature=0)

    #Load docs with metadata
    all_docs = load_documents_with_metadata(folder_path, llm_for_loader)

    if not all_docs:
        raise ValueError(f"No documents found in {folder_path}")
    
    #Filter out documents already in vector store
    existing_hashes = get_existing_hashes(vectordb)
    new_docs = filter_new_hashes(all_docs, existing_hashes)

    if new_docs:
        splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(new_docs)
        vectordb.add_documents(chunks)
        print(f"Added {len(chunks)} new chunks to vector store")
    else:
        print("No new documents added")

    #LLM
    llm = OpenAI(
        temperature=0, 
        api_key=openai_api_key,
        model_name='gpt-4o-mini')
    
    #Build retreiver
    base_retriever = vectordb.as_retriever()
    compressor = LLMChainExtractor.from_llm(llm)

    retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        document_compressor=compressor
    )


    #Create template
    prompt = PromptTemplate.from_template(
        "Answer the question based only on the context below:\n\n{context}\n\nQuestion: {question}"
        "Be concise with the answer"
    )   

    

    #Build RAG chain
    rag_chain = (
        RunnableParallel({
            "context": retriever | (lambda docs: format_docs_token_limited(docs, model_name='gpt-4o-mini')),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



