import os
import streamlit as st
import pickle


def load_retriever(cache_path, build_retriever, *args, **kwargs):
    if os.path.exists(cache_path):
        retriever = load_retriever_from_cache(cache_path)
    else:
        retriever = build_retriever(*args, **kwargs)
        with open(cache_path, "wb") as f:
            pickle.dump(retriever, f)
    return retriever    

@st.cache_resource
def load_retriever_from_cache(cache_path):
    with open(cache_path, "rb") as f:
        return pickle.load(f)
      