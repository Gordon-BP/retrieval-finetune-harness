from typing import List, Tuple
from os import PathLike
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import VectorStoreRetriever

class OpenAIChain():
    def __init__(self, 
                 model:str = "gpt-3.5-turbo-instruct",
                 index_dir:PathLike = "./indices/en"):
        # Assert that the env OPENAI_API_KEY is set
        self.model = model
        embeddings = OpenAIEmbeddings()
        print("Hello")

    def __call__(language:str,
                 history:List[Tuple[str]], 
                 retriever:VectorStoreRetriever)-> str:
        
        return "Hey this is OpenAI"