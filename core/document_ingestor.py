import os
import hashlib
from langchain_community.document_loaders import UnstructuredFileLoader
from core.loader_manager import LoaderManager

class DocumentIngestion:
    def __init__(self, folder_path, loader_agent, vectorstore):
        self.folder_path = folder_path
        self.loader_agent = loader_agent
        self.vectorstore = vectorstore

    def ingest_new_documents(self):
        all_docs = self.load_documents_with_metadata()
        existing_hashes = self.get_existing_hashes()
        return self.filter_new_hashes(all_docs, existing_hashes)
      
    def load_documents_with_metadata(self):
        all_docs=[]

        for filename in os.listdir(self.folder_path):
            path = os.path.join(self.folder_path,filename)
            if not os.path.isfile(path):
                continue

            preview = self.get_preview(path)
            loader_name = self.loader_agent.choose_loader(filename, preview)
            loader_class = LoaderManager.loader_mapping.get(loader_name, UnstructuredFileLoader)

            try:
                loader =loader_class(path)
                file_hash = self.hash_file(path)
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata["filename"] = filename
                    doc.metadata["file_hash"] = file_hash

                all_docs.extend(docs)
                print(f"✅ Loaded {len(docs)} documents from {filename}")
                
            except Exception as e:
                print(f" Error processing {filename}: {e}")

        return all_docs
    
    def get_preview(self, path):
        try:
            with open(path, "r", encoding='utf-8', errors='ignore') as f:
                return f.read(1000)
        except Exception as e:
            print(f"Failed to read preview for {path}: {e}")
            return "[unreadable preview]"
    
    def hash_file(self, path):
        #Return an MDS hash of a file
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def get_existing_hashes(self):
        #Extract existing file hashs from vector store metadata
        try:
            existing = self.vectorstore.get(include=["metadatas"])
            return {meta["file_hash"] for meta in existing["metadatas"] if "file_hash" in meta}
        except Exception as e:
            print(f"Failed to get existing hashes: {e}")
            return set()
        
    def filter_new_hashes(self, all_docs, existing_hashes):
        #Filter out hashes that already exist in vector store
        return [doc for doc in all_docs if doc.metadata.get("file_hash") not in existing_hashes]
        