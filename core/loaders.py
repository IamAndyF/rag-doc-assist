from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, UnstructuredWordDocumentLoader, 
    UnstructuredFileLoader)

loader_mapping = {
    "PyPDFLoader": PyPDFLoader,
    "TextLoader": TextLoader,
    "CSVLoader": CSVLoader,
    "UnstructuredWordDocumentLoader": UnstructuredWordDocumentLoader,
    "UnstructuredFileLoader": UnstructuredFileLoader  #Fallback file loader
}

valid_loaders = set(loader_mapping.keys())

extension_loader_map = {
    ".pdf": "PyPDFLoader",
    ".txt": "TextLoader",
    ".csv": "CSVLoader",
    ".docx": "UnstructuredWordDocumentLoader"
}