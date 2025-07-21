import os
from langchain.prompts import PromptTemplate

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

    response = llm.invoke(prompt)
    loader_name = response.strip()
    return loader_name