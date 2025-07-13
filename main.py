import streamlit as st
from core.config import load_config
from core.rag_engine import build_rag_engine

config = load_config()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_rag_engine(
        openai_api_key=config["openai_api_key"],
        folder_path="/users/andyfung/ai/doc_assist/data",
        persist_dir=config['persist_dir']
    )

st.title("📄 Doc Assist — AI Knowledge Assistant")

query = st.text_input("Ask a question based on internal documents:")

if query:
    result = st.session_state.qa_chain.invoke(query)
    st.markdown("Answer:")
    st.write(result)

    #Show sources
    st.markdown("Sources Used:")
    if result["docs"]:
        for i, doc in enumerate(result["docs"], 1):
            filename = doc.metadata.get("filename", "Unknown file")
            snippet = doc.page_content[:300].strip().replace("\n", " ")
            st.markdown(f"**{i}. {filename}**")
            st.markdown(f"> {snippet}...")
    else:
        st.write("No source documents returned")
