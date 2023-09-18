# Class for the cross encoder finetune and eval pipeline
from torch.nn import CrossEntropyLoss

# @title Re-rank with cross-encoder
# @title BiEncoderEvalPipeine
import hnswlib
import torch
from sentence_transformers import SentenceTransformer, util


class ReRankEvalPipeline:
    """
    Test harness for fine-tuned bi-encoder models

    Args:
    - bi_encoder (SentenceTransformer): The fine-tuned model
    - corpus ([str]): List of docs that get retrieved per query
    """

    # TODO: Make this take a biencoder pipeline so we don't have to rewrite the tests
    def __init__(
        self,
        bi_encoder,
        cross_encoder,
        corpus,
    ):
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        if torch.cuda.is_available():
            self.bi_encoder.to(device="cuda")

        self.corpus = corpus
        self.corpus_embeddings = self.bi_encoder.encode(
            corpus, convert_to_tensor=True, device="cuda"
        )
        self.index = None

    def _search_hnsw(self, query_embedding, k):
        if self.index is None:
            dim = self.corpus_embeddings.shape[1]
            self.index = hnswlib.Index(space="cosine", dim=dim)
            self.index.init_index(
                max_elements=len(self.corpus), ef_construction=100, M=16
            )
            self.index.add_items(self.corpus_embeddings.cpu().numpy())
            self.index.set_ef(50)

        labels, distances = self.index.knn_query(query_embedding.cpu().numpy(), k=k)
        return labels[0]

    def _search(self, query_embedding, method, k):
        if method == "cosine":
            cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=k)
            return top_results.indices.cpu()

        elif method == "dot":
            dot_scores = util.dot_score(query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(dot_scores, k=k)
            score = top_results[0]
            return top_results.indices.cpu()

        elif method == "euclidean":
            distances = torch.norm(query_embedding - self.corpus_embeddings, dim=1)
            top_results = torch.topk(
                -distances, k=k
            )  # Negative because we want smallest distances
            score = top_results[0]
            return top_results.indices.cpu()

        elif method == "hnsw":
            return self._search_hnsw(query_embedding, k=k)

        else:
            raise ValueError(f"Unsupported search method: {method}")

    def search(self, query, k, method="cosine"):
        # Bi-encoder search
        query_embedding = self.bi_encoder.encode(
            query, convert_to_tensor=True, device="cuda"
        )
        top_hits = self._search(query_embedding, method, k)

        # Cross-encoder re-ranking
        pairs = [(query, self.corpus[idx]) for idx in top_hits]
        scores = self.cross_encoder.predict(pairs)
        # Filter based on threshold
        results = [
            self.corpus[top_hits[i]]
            for i, score in enumerate(scores)
            if score > CROSS_ENCODER_THRESHOLD
        ]
        if len(results) == 0:
            return UNKNOWN_PHRASE
        else:
            # Sort results by the cross-encoder scores
            # final_results = [results[i][0] for i in range(len(results)) if i<=2]
            return results[0]
