import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def load_documents_from_folder(folder_path):
    all_docs=[]
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())
        elif filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            loader = TextLoader(path)
            all_docs.extend(loader.load())
    return all_docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_engine(openai_api_key, folder_path, persist_dir="./chroma_store"):
    #Load and split docs
    documents = load_documents_from_folder(folder_path)

    if not documents:
        raise ValueError(f'No documents found in {folder_path}')
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    #Embed and store
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=openai_api_key)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

    #Build retreiver
    retriever = vectordb.as_retriever()

    #Create template
    prompt = PromptTemplate.from_template(
        "Answer the question based only on the context below:\n\n{context}\n\nQuestion: {question}"
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



