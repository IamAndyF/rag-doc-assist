import streamlit as st
from core.config import load_config
from core.rag_engine import RAGEngine

config = load_config()

rag_engine = RAGEngine(
    openai_api_key=config["openai_api_key"],
    folder_path="/users/andyfung/ai/doc_assist/data",
    persist_dir=config['persist_dir']
)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = rag_engine.build_rag_chain()

st.title("📄 Doc Assist — AI Knowledge Assistant")

query = st.text_input("Ask a question based on internal documents:")

if query:
    result = st.session_state.qa_chain.invoke(query)
    st.markdown("Answer:")
    st.write(result)

