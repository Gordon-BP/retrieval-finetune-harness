# @title 1. Transform Dataset to JPQ-friendly format
from beir import util
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader

import pathlib, os, csv
import argparse
import logging
import random

random.seed(42)

#### Just some code to print debug information to stdout
logger = logging.getLogger(__name__)


class DataTransformer:
    def preprocessing(text):
        return text.replace("\r", " ").replace("\t", " ").replace("\n", " ").strip()

    def transform(
        dataset: str,
        output_dir: str,
        prefix: str = None,
        beir_data_root: str = None,
        split: str = "train",
    ):
        if not beir_data_root:
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
                dataset
            )
            out_dir = os.path.join(pathlib.Path(".").parent.absolute(), "datasets")
            data_path = util.download_and_unzip(url, out_dir)
        else:
            data_path = os.path.join(beir_data_root, dataset)

        if prefix:
            corpus, queries, qrels = GenericDataLoader(data_path, prefix=prefix).load(
                split=split
            )
        else:
            corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

        corpus_ids, query_ids = list(corpus), list(queries)
        doc_map, query_map = {}, {}

        for idx, corpus_id in enumerate(corpus_ids):
            doc_map[corpus_id] = idx

        for idx, query_id in enumerate(query_ids):
            query_map[query_id] = idx

        os.makedirs(output_dir, exist_ok=True)

        print(
            "Writing Corpus to file: {}...".format(
                os.path.join(output_dir, "collection.tsv")
            )
        )
        with open(
            os.path.join(output_dir, "collection.tsv"), "w", encoding="utf-8"
        ) as fIn:
            writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for doc_id in tqdm(corpus_ids, total=len(corpus_ids)):
                doc = corpus[doc_id]
                doc_id_new = doc_map[doc_id]
                writer.writerow(
                    [
                        doc_id_new,
                        preprocessing(doc.get("title", ""))
                        + " "
                        + preprocessing(doc.get("text", "")),
                    ]
                )

        print(
            "Writing Queries to file: {}...".format(
                os.path.join(output_dir, "queries.tsv")
            )
        )
        with open(
            os.path.join(output_dir, "queries.tsv"), "w", encoding="utf-8"
        ) as fIn:
            writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for qid, query in tqdm(queries.items(), total=len(queries)):
                qid_new = query_map[qid]
                writer.writerow([qid_new, preprocessing(query)])

        print(
            "Writing Qrels to file: {}...".format(
                os.path.join(output_dir, "qrels.train.tsv")
            )
        )
        with open(
            os.path.join(output_dir, "qrels.train.tsv"), "w", encoding="utf-8"
        ) as fIn:
            writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for qid, docs in tqdm(qrels.items(), total=len(qrels)):
                for doc_id, score in docs.items():
                    qid_new = query_map[qid]
                    doc_id_new = doc_map[doc_id]
                    writer.writerow([qid_new, 0, doc_id_new, score])


dt = DataTransformer()
dt.transform(
    dataset="nfcorpus",
    output_dir="./datasets/nfcorpus",
    prefix="",
)
