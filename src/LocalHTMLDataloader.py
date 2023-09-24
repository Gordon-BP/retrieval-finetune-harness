from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dataclasses import dataclass
from typing import List
import os
from os import PathLike

@dataclass
class LocalHTMLDataloader:
    data_dir: PathLike = "./data"
    chunk_size: int = 350
    chunk_overlap: int = 55
    @staticmethod
    def load_html(file_path: PathLike) -> List[Document]:
        """
        Yes, this is barely enough code to justify its own method.
        I'm leaving it like this so that, in the future, if I want
        to do some funky shit with HTML files I can do it here.
        """
        return UnstructuredHTMLLoader(file_path, 
                                      mode="elements").load()

    def process_dir(self, dir: PathLike) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap  = self.chunk_overlap,
            length_function = len,
            add_start_index = True,
        )
        chunks = []
        for filename in os.listdir(dir):
            if filename.endswith('.html'):
                file_path = os.path.join(dir, filename)
                docs = self.load_html(file_path) 
                chunks.append(text_splitter.split_documents(docs))
        return chunks