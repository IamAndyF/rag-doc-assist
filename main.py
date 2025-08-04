import os
import streamlit as st
from core.rag_engine import RAGEngine
from core.config import OPENAI_API_KEY, PERSIST_DIR, DATA_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME, EMBEDDING_MODEL


# Set streamlit page config
st.set_page_config(page_title="Doc Assist", layout="centered")
st.title("Doc Assist - AI Knowledge Assistant")

# Initialise RAG engine
rag_engine = RAGEngine(
    api_key=OPENAI_API_KEY,
    persist_dir=PERSIST_DIR,
    folder_path=DATA_FOLDER,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    model=MODEL_NAME,
    embedding_model=EMBEDDING_MODEL
)

# Create upload mechanism
st.subheader("Document upload")
uploaded_files = st.file_uploader(
    "Upload new documents to ingest",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DATA_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
            st.success(f"Uploaded {file.name} successfully!")

    if st.button("Ingest documents"):
        with st.spinner("Processing uploaded documents..."):
            rag_engine.ingest_new_documents()


st.divider()

# RAG chain initialisation
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = rag_engine.build_rag_chain()

st.subheader("Query search")
query = st.text_input("Ask a question based on internal documents:")

if query:
    with st.spinner("Retrieving answer..."):
        result = st.session_state.qa_chain.invoke(query)

    st.markdown("Answer:")
    st.write(result.answer)
    st.write("Sources:")
    for s in result.sources:
        st.write(f"-{s}")