import pandas as pd
import re
import os
import aiohttp
import asyncio
from typing import List
from src.TransformerEmbeddings import TransformerEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, AutoModel

class EmbeddingGenerator:
    def __init__(self, docs_path: str):
        """
        Initialize the EmbeddingGenerator.

        :param docs_path: Path to the csv_file where your data is stored
        """
        self.articles_path = docs_path
        load_dotenv()
        self.semaphore = asyncio.Semaphore(2)
        self.pbar = None
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.cohere_api_key = os.environ["COHERE_API_KEY"]

    def load_chunks(
        self, chunk_size: int = 400, chunk_overlap: int = 40
    ) -> List[Document]:
        """
        Load article data and split it into chunks.

        :param chunk_size: Size of each chunk in tokens.
        :param chunk_overlap: Overlapping size between chunks (also in tokens).
        :return: List of document chunks.
        """
        df_all = pd.read_csv(self.articles_path)
        loader = DataFrameLoader(df_all, page_content_column="content")
        data = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents(data)

    @staticmethod
    def save_to_tsv(embeddings_batch, filename="embeddings.tsv") -> None:
        """
        Save embeddings to a TSV file.

        :param embeddings_batch: List of embeddings.
        :param filename: File name to save embeddings.
        """
        with open(filename, "a", encoding="utf-8") as file:  # 'a' denotes append mode
            for line in embeddings_batch:
                file.write("\t".join(map(str, line)) + "\n")

    async def async_embed_cohere(
        self, session, page_content, metadata: List[str]
    ) -> List[tuple]:
        """
        Asynchronously generate embeddings using Cohere API.

        :param session: aiohttp client session.
        :param page_content: List of text to be embedded.
        :param metadata: Associated metadata.
        :return: List of embedded data.
        """
        api_url = "https://api.cohere.ai/v1/embed"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.cohere_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "texts": page_content,
            "model": "embed-multilingual-v2.0",
            "truncate": "END",
        }

        async with session.post(api_url, headers=headers, json=data) as response:
            async with self.semaphore:
                try:
                    if response.status == 200:
                        result = await response.json()
                        embeddings = result["embeddings"]
                        EmbeddingGenerator.save_to_tsv(
                            list(zip(page_content, embeddings, metadata)),
                            filename="cohere_embeddings3.csv",
                        )
                        return list(zip(page_content, embeddings))
                    if response.status == 429:
                        print(
                            "Hit Cohere API rate limit, waiting 10 seconds before retrying..."
                        )
                        await asyncio.sleep(10)
                        return await self.async_embed_cohere(
                            session, page_content, metadata
                        )
                except Exception as e:
                    print(f"There was an error: {e}")

    async def get_cohere_embeddings(self, chunks, batch_size: int = 95) -> List[tuple]:
        """
        Generate embeddings using Cohere for given chunks.

        :param chunks: List of text chunks.
        :param batch_size: Size of each batch to be processed.
        :return: List of embeddings.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(0, len(chunks), batch_size):
                page_content, segment_metadata = zip(*chunks[i : i + 95])
                metadata = [data[-1] for data in segment_metadata]
                if len(page_content) > batch_size:
                    raise ValueError(
                        f"Size of batch ({len(page_content)}) is bigger than specified limit of {batch_size}"
                    )

                await asyncio.sleep(0.7)  # Sleep for a bit to not hit rate limit

                async with self.semaphore:
                    task = asyncio.create_task(
                        self.async_embed_cohere(session, page_content, metadata)
                    )
                    tasks.append(task)
                    self.pbar.update(1)

            results = await asyncio.gather(*tasks)
            embeddings = [i for i in results]
            return embeddings

    def generate_embeddings_langchain(
        self, index_name: str = "openai-embeddings"
    ) -> None:
        """
        Generate embeddings using Langchain OpenAI implementation and save to a FAISS index.

        :param index_name: Name of the FAISS index file.
        """
        chunks = self.load_chunks()
        # # Embed with OpenAI
        oai_embed = OpenAIEmbeddings(show_progress_bar=True, max_retries=60000000)
        print("Embedding with OpenAI...")
        db1 = FAISS.from_documents(chunks, oai_embed)
        db1.save_local(index_name)

    def generate_embeddings_cohere(self, index_name: str = "cohere-embeddings") -> None:
        """
        Generate embeddings using Cohere and save them to a FAISS index.

        :param index_name: Name of the FAISS index file.
        """

        def prep_text(text):
            return re.sub("\n|\t", "", text[-1]).strip()

        chunks = [
            [prep_text(page_content), metadata] for page_content, metadata in chunks
        ]
        print("Embedding with Cohere...")
        metadata = [metadata[-1] for page_content, metadata in chunks]
        loop = asyncio.get_event_loop()
        self.pbar = tqdm(total=len(chunks), desc="Embedding with Cohere")
        embeds = loop.run_until_complete(self.get_cohere_embeddings(chunks))
        texts = []
        texts.extend([doc for batch in embeds for doc in batch])
        db2 = FAISS.from_embeddings(
            text_embeddings=texts,
            embedding=CohereEmbeddings(model="embed-multilingual-v2.0"),
            metadatas=metadata,
        )
        print("All done! Saving...")
        db2.save_local(index_name)
    
    def generate_embeddings_hf(self, model_name:str="bert-base-multilingual-cased", index_name:str="bert-base-multilingual-cased-embeddings")->None:
        # Preprocess text
        chunks = self.load_chunks()
        def prep_text(text)->str:
            return re.sub("\n|\t", "", text[-1]).strip()
        contents, metadata = zip(*
            [(prep_text(page_content), metadata[-1]) for page_content, metadata in chunks]
            )
        # Load model from HuggingFace Hub
        model = TransformerEmbeddings(model_name)
        # embed stuff
        doc_emb = model.embed_documents(contents)

        db = FAISS.from_embeddings(
            text_embeddings=doc_emb,
            embedding=model,
            metadatas=metadata,
        )
        print("All done! Saving...")
        db.save_local(index_name)