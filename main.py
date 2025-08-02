import streamlit as st
from core.rag_engine import RAGEngine
from core.config import OPENAI_API_KEY, PERSIST_DIR, DATA_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME, EMBEDDING_MODEL

rag_engine = RAGEngine(
    api_key=OPENAI_API_KEY,
    persist_dir=PERSIST_DIR,
    folder_path=DATA_FOLDER,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    model=MODEL_NAME,
    embedding_model=EMBEDDING_MODEL
)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = rag_engine.build_rag_chain()

st.title("📄 Doc Assist — AI Knowledge Assistant")

query = st.text_input("Ask a question based on internal documents:")

if query:
    result = st.session_state.qa_chain.invoke(query)
    st.markdown("Answer:")
    st.write(result)

