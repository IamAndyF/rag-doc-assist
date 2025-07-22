🧠 rag-doc-assist: Retrieval-Augmented Generation (RAG) Knowledge Assistant

Overview

rag-doc-assist is an AI-powered knowledge assistant that allows you to query your own documents using the power of large language models. It leverages Retrieval-Augmented Generation (RAG) to provide concise, context-aware answers from documents stored in a local folder.

This project uses LangChain, OpenAI, and Chroma for vector storage, along with an intelligent AI agent that automatically selects the appropriate loader for each document based on its content.

✨ Features

✅ Drop-in folder-based document ingestion

✅ AI agent intelligently determines document type and chooses optimal loader (PDF, TXT, DOCX, etc.)

✅ Deduplication and vectorization via OpenAIEmbeddings

✅ Token-based document chunking

✅ Customizable prompt templates for querying

✅ Built with Streamlit for interactive use

🧱 Architecture

Loader Agent: Uses an LLM to preview document content and select the best LangChain loader

Embedding & Vector Store: Uses OpenAIEmbeddings and Chroma

Chunking: Token-based chunking with overlap for coherent retrieval

RAG Flow: Vector database returns top relevant chunks -> fed into LLM via prompt -> generated answer