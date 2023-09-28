import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List
from langchain.schema import Document
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline


class QueryGenerator:
    """
    Generates queries from documents using various models based on the document language.
    """
    
    class ParagraphDataset(Dataset):
        """Dataset for document chunks."""
        def __init__(self, dataframe):
            self.dataframe = dataframe

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            return row['page_content'], row['language'], row['id']

    @staticmethod
    def save_to_tsv(data, filename="docs-with-queries.tsv"):
        """Saves data lines to a TSV file."""
        with open(filename, "a", encoding="utf-8") as file:
            for line in data:
                file.write("\t".join(map(str, line)) + "\n")

    @staticmethod
    def create_queries(batched_data, tokenizer, model, device, n=5):
        """Generates queries from a batch of document chunks."""
        batched_data, batched_lang, batched_id = batched_data
        input_ids = tokenizer(batched_data, return_tensors='pt', padding=True, truncation=False).input_ids.to(device)

        with torch.no_grad():
            beam_outputs = model.generate(
                input_ids=input_ids,
                max_length=150,
                num_beams=8,
                no_repeat_ngram_size=5,
                num_return_sequences=n,
                early_stopping=True
            )
        
        for data, lang, doc_id, beam_output in zip(batched_data, batched_lang, batched_id, beam_outputs):
            i = n
            for output in beam_outputs:
                query = tokenizer.decode(output, skip_special_tokens=True)
                QueryGenerator.save_to_tsv([(doc_id.item(), i%n, lang, query)], filename=f"new-{lang}-queries.tsv")
                i+=1
        #clean up
        del query
        del input_ids
        del beam_outputs
        gc.collect()


    def __init__(self, path_to_csv, batch_size=16, num_workers=4):
        """
        Initializes the QueryGenerator.

        :param path_to_csv: Path to the CSV containing document data.
        :param batch_size: Batch size for DataLoader.
        :param num_workers: Number of workers for DataLoader.
        """
        self.model_dict = {
            "ar":"doc2query/msmarco-arabic-mt5-base-v1",
            "bn":None,
            "en-us":"doc2query/msmarco-t5-base-v1",
            "es":"doc2query/msmarco-spanish-mt5-base-v1",
            "fr":"doc2query/msmarco-french-mt5-base-v1",
            "hi":"doc2query/msmarco-hindi-mt5-base-v1",
            "ja":"doc2query/msmarco-japanese-mt5-base-v1",
            "ko":None,
            "pt-br":"doc2query/msmarco-portuguese-mt5-base-v1",
            "sw":None,
            "th":None,
            "ur":None,
            "vi":"doc2query/msmarco-vietnamese-mt5-base-v1"
        }
        self.df = self._load_data(path_to_csv)
        self.dataloaders = self._create_dataloaders(batch_size, num_workers)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_data(self, path):
        """Load chunks from a CSV file and convert to DataFrame."""
        chunks = self._load_chunks(path)
        chunk_records = [{**doc.metadata, "page_content": doc.page_content} for doc in chunks]
        return pd.DataFrame.from_records(chunk_records)

    @staticmethod
    def _load_chunks(path, chunk_size: int = 400, chunk_overlap: int = 40) -> List[Document]:
        """Load article data from a CSV and split it into chunks."""
        loader = DataFrameLoader(pd.read_csv(path), page_content_column="content")
        data = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents(data)
    
    def _create_dataloaders(self, batch_size, num_workers):
        """Create data loaders grouped by language."""
        dataloaders = {}
        grouped = self.df.groupby('language')
        for lang, group in grouped:
            dataset = self.ParagraphDataset(group)
            dataloaders[lang] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return dataloaders

    def process_documents(self, skip_languages:List[str]=[]):
        """Processes documents to generate queries."""
        for lang, dataloader in self.dataloaders.items():
            model_name = self.model_dict.get(lang)
            if model_name is None:
                print(f"Skipping {lang} as no model is available.")
                continue

            if lang in skip_languages:
                print(f"Skipping {lang}...")

            if lang == "en-us": #English is special and it gets its own T5
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

            for batch in tqdm(dataloader, desc=f"Processing batches for {lang}"):
                self.create_queries(batch, tokenizer, model, self.device)
                
            # Clean up memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()