import os

import streamlit as st

from core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_FOLDER,
    EMBEDDING_MODEL,
    MODEL_NAME,
    OPENAI_API_KEY,
    PERSIST_DIR,
    CACHE_PATH
)
from core.rag_engine import LLMClient, RAGEngine
from core.sql_agent import SQLAgent

# Set streamlit page config
st.set_page_config(page_title="Doc Assist", layout="centered")
st.title("Doc Assist - AI Knowledge Assistant")


# Initialise RAG engine
@st.cache_resource
def get_rag_engine():
    return RAGEngine(
        api_key=OPENAI_API_KEY,
        persist_dir=PERSIST_DIR,
        folder_path=DATA_FOLDER,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        model=MODEL_NAME,
        embedding_model=EMBEDDING_MODEL,
        cache_path=CACHE_PATH
    )


rag_engine = get_rag_engine()


# Initialise SQL agent
@st.cache_resource
def get_sql_agent():
    llm_client = LLMClient(rag_engine.llm)
    return SQLAgent(llm_client)


sql_agent = get_sql_agent()

if "sql_agent" not in st.session_state:
    llm_client = LLMClient(rag_engine.llm)
    st.session_state.sql_agent = SQLAgent(llm_client)


with st.sidebar:
    # Create upload mechanism
    st.subheader("Document upload")
    uploaded_files = st.file_uploader(
        "Upload new documents to ingest",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(DATA_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                st.success(f"Uploaded {file.name} successfully!")


# RAG chain initialisation
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = rag_engine.build_rag_chain()

st.subheader("Query search")

with st.form("query_form"):
    mode = st.radio("Select mode:", ["Document", "Database"], horizontal=True)
    query = st.text_input(
        "Ask a question", placeholder="Ask a question based on internal documents:"
    )
    submitted = st.form_submit_button("Search")

if submitted and query.strip():
    with st.spinner("Retrieving answer..."):
        if mode == "Document":
            result = st.session_state.qa_chain.invoke(query)
            st.markdown("Answer:")
            st.write(result)
        else:  # Database mode
            result = st.session_state.sql_agent.run_query(query)
            st.markdown("Answer:")
            st.write(result)
