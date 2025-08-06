from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader,
)


class LoaderManager:
    loader_mapping = {
        "PyPDFLoader": PyPDFLoader,
        "TextLoader": TextLoader,
        "CSVLoader": CSVLoader,
        "UnstructuredWordDocumentLoader": UnstructuredWordDocumentLoader,
        "UnstructuredFileLoader": UnstructuredFileLoader,  # Fallback file loader
    }

    valid_loaders = set(loader_mapping.keys())

    extension_loader_map = {
        ".pdf": "PyPDFLoader",
        ".txt": "TextLoader",
        ".csv": "CSVLoader",
        ".docx": "UnstructuredWordDocumentLoader",
    }

    @classmethod
    def is_valid_loader(cls, loader_name):
        return loader_name in cls.valid_loaders

    @classmethod
    def get_loader_by_extension(cls, extension):
        return cls.extension_loader_map.get(extension, "UnstructuredFileLoader")
