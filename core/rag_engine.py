import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from core.loader_utils import load_documents_with_metadata, get_existing_hashes, filter_new_hashes

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

    #Build retreiver
    retriever = vectordb.as_retriever()

    #Create template
    prompt = PromptTemplate.from_template(
        "Answer the question based only on the context below:\n\n{context}\n\nQuestion: {question}"
        "Be concise with the answer"
    )   

    #LLM
    llm = OpenAI(
        temperature=0, 
        api_key=openai_api_key,
        model_name='gpt-4o-mini')

    #Build RAG chain
    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



