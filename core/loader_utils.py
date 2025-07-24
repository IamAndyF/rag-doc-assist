import os
import hashlib
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate

from core.ai_agents import loader_agent
from core.loaders import loader_mapping


def hash_file(path):
    #Return an MDS hash of a file
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    
def load_documents_with_metadata(folder_path, llm):
    all_docs=[]

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path,filename)
        if not os.path.isfile(path):
            continue

        try:
            #Try to preview for text based files
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                preview = f.read(1000)
        except Exception:
            preview = "[binary or unreadable preview]"

        try:
            #Use loader ai agent to determine the file loader type
            loader_name = loader_agent(filename, preview, llm)
            loader_class = loader_mapping.get(loader_name, UnstructuredFileLoader)

            if loader_name not in loader_mapping:
                print(f"Unrecognised loader {loader_name}, defaulting to UnstructuredFileLoader")
            
            #Get file hash and load documents
            loader =loader_class(path)
            file_hash = hash_file(path)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["filename"] = filename
                doc.metadata["file_hash"] = file_hash

            print(f"✅ Loaded {len(docs)} documents from {filename}")
            all_docs.extend(docs)

        except Exception as e:
            print(f" Error processing {filename}: {e}")

    return all_docs

def get_existing_hashes(vectordb):
    #Extract existing file hashs from vector store metadata
    try:
        existing = vectordb.get(include=["metadatas"])
        return {meta["file_hash"] for meta in existing["metadatas"] if "file_hash" in meta}
    except Exception as e:
        print(f"Failed to get existing hashes: {e}")
        return set()
    
def filter_new_hashes(all_docs, existing_hashes):
    #Filter out hashes that already exist in vector store
    return [doc for doc in all_docs if doc.metadata.get("file_hash") not in existing_hashes]
    
