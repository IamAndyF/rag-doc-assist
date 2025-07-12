import os
import hashlib
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, UnstructuredWordDocumentLoader, 
    UnstructuredFileLoader)

def hash_file(path):
    #Return an MDS hash of a file
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_documents_with_metadata(folder_path):
    all_docs=[]

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path,filename)
        file_ext = os.path.splitext(filename)[1].lower()

        try:
            #Select loader based on file extension
            if file_ext == ".pdf":
                loader = PyPDFLoader(path)
            elif file_ext == ".txt":
                loader = TextLoader(path)
            elif file_ext == ".docx":
                loader = UnstructuredWordDocumentLoader(path)
            elif file_ext == ".csv":
                loader = CSVLoader(path)
            else:
                loader = UnstructuredFileLoader(path)
            
            #Get file hash and load documents
            file_hash = hash_file(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["filename"] = filename
                doc.metadata["file_hash"] = file_hash

            print(f"✅ Loaded {len(docs)} documents from {filename}")
            all_docs.extend(loader.load())

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
    
