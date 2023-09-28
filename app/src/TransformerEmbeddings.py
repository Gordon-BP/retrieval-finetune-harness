from langchain.schema.embeddings import Embeddings
import torch
import asyncio
from transformers import AutoTokenizer, AutoModel
from typing import List

class TransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """
        Initialize TransformerEmbeddings with a specific transformer model.
        
        :param model_name: HuggingFace model identifier or path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts using the transformer model.
        
        :param texts: List of texts to embed.
        :return: List of embeddings.
        """
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text using the transformer model.
        
        :param text: Query text to embed.
        :return: Embedding for the query.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embed multiple texts using the transformer model.
        
        :param texts: List of texts to embed.
        :return: List of embeddings.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronously embed a single query text using the transformer model.
        
        :param text: Query text to embed.
        :return: Embedding for the query.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

