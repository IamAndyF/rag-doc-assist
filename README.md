Retrieval-Augmented Generation (RAG) Knowledge Assistant

A simple yet scalable prototype for a RAG-based document Q&A assistant using OpenAI models and Chroma vector store.

core/
  ├── rag_engine.py         # RAG pipeline logic
  ├── document_ingestor.py  # Ingestion & deduplication
  ├── loader_agent.py       # LLM-based loader selector
  ├── loader_manager.py     # Fallback loader map
  ├── llm_client.py         # LLM wrapper
utils/
  ├── hashing_utils.py      # File preview & hashing
app/
  ├── app.py                # Streamlit frontend
.env                        # Config

To run, install dependencies from requirements.txt

Setup .env file with openai_key or equivalent: 
OPENAI_API_KEY=your-openai-key

To run use: streamlit run main.py


