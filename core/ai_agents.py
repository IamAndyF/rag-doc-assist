import os
from langchain.prompts import PromptTemplate
from core.loaders import valid_loaders, extension_loader_map

def loader_agent(filename, preview, llm): 
    #AI agent decides the best loader to use
    extension = os.path.splitext(filename)[1].lower()

    Prompt_Template = PromptTemplate.from_template(
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

    prompt = Prompt_Template.format(
        filename=filename,
        extension=extension,
        preview=preview[:1000]
    )
    
    try:
        response = llm.invoke(prompt)
        loader_name = response.strip()

        # Validate the loader name
        if loader_name not in valid_loaders:
            print(f"Unrecognised loader {loader_name}, falling back to extentions mapping")
            return get_fallback_loader(extension)

        return loader_name
    
    except Exception as e: 
        print(f'LLM error: {e}, using fallback loader')
        return get_fallback_loader(extension)
    
def get_fallback_loader(extension):
    # Fallback loader based on file extension, if not then default to UnstructuredFileLoader
    return extension_loader_map.get(extension, "UnstructuredFileLoader")
    