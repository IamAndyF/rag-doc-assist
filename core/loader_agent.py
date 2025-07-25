import os
from langchain.prompts import PromptTemplate
from core.loader_manager import LoaderManager

class LoaderAgent:
    def __init__(self, llm, valid_loaders, extension_loader_map):
        self.llm = llm
        self.valid_loaders = valid_loaders
        self.extension_loader_map = extension_loader_map

        self.prompt_template = PromptTemplate.from_template(
            """You are an intelligent assistant that selects the best document loader for a file based on its filename, file extension, and a preview of its contents (if readable).
            
            Choose one of the following loaders:

            - PyPDFLoader
            - TextLoader
            - CSVLoader
            - UnstructuredWordDocumentLoader
            - UnstructuredFileLoader

            Only respond with one exact loader name (case-sensitive).

            ### File Info ###
            Filename: {filename}
            Extension: {extension}
            Preview: {preview}
            """
        )

    def choose_loader(self, filename, preview):
        extension = os.path.splitext(filename)[1].lower()
        
        prompt = self.prompt_template.format(
            filename=filename,
            extension=extension,
            preview=preview[:1000]
        )
        
        try:
            response = self.llm.invoke(prompt)
            loader_name = response.strip()

            # Validate the loader name
            if loader_name not in self.valid_loaders:
                print(f"Unrecognised loader {loader_name}, falling back to extentions mapping")
                return self.get_fallback_loader(extension)

            return loader_name
        
        except Exception as e: 
            print(f'LLM error: {e}, using fallback loader')
            return self.get_fallback_loader(extension)
        
    def get_fallback_loader(self, extension):
        # Fallback loader based on file extension, if not then default to UnstructuredFileLoader
        return LoaderManager.get_loader_by_extension(extension)
        