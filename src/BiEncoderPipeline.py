# Class for the bi-encoder train & eval pipeline
import hnswlib
import torch
import torch.nn.functional as F


class BiEncoderEvalPipeline:
    """
    Test harness for fine-tuned bi-encoder models

    Args:
    - bi_encoder (SentenceTransformer): The fine-tuned model
    - corpus ([str]): List of docs that get retrieved per query
    """

    def __init__(
        self,
        bi_encoder,
        corpus,
        COSINE_THRESHOLD,
        DOT_THRESHOLD,
        EUC_THRESHOLD,
        HNSW_THRESHOLD,
        k,
    ):
        self.bi_encoder = bi_encoder
        if torch.cuda.is_available():
            self.bi_encoder.to(device="cuda")
        self.corpus = corpus
        self.corpus_embeddings = self.bi_encoder.encode(
            corpus, convert_to_tensor=True, device="cuda"
        )
        self.index = None
        (
            self.COSINE_THRESHOLD,
            self.DOT_THRESHOLD,
            self.EUC_THRESHOLD,
            self.HNSW_THRESHOLD,
        ) = (COSINE_THRESHOLD, DOT_THRESHOLD, EUC_THRESHOLD, HNSW_THRESHOLD)
        self.k = k

    def _search_hnsw(self, query_embedding):
        if self.index is None:
            dim = self.corpus_embeddings.shape[1]
            self.index = hnswlib.Index(space="cosine", dim=dim)
            self.index.init_index(
                max_elements=len(self.corpus), ef_construction=100, M=16
            )
            self.index.add_items(self.corpus_embeddings.cpu().numpy())
            self.index.set_ef(50)

        labels, distances = self.index.knn_query(
            query_embedding.cpu().numpy(), k=self.k
        )
        docs = [
            labels[i][0] if distances[i][0] <= self.HNSW_THRESHOLD else None
            for i in range(len(labels))
        ]
        return docs

    def _search(self, query_embedding, method):
        if method == "cosine":
            cos_scores = F.cosine_similarity(query_embedding, self.corpus_embeddings)
            top_results = torch.topk(cos_scores, k=self.k)
            hits = [
                res.indices.cpu() if res >= self.COSINE_THRESHOLD else None
                for res in top_results
            ]
            return hits

        elif method == "dot":
            dot_scores = torch.mm(query_embedding, self.corpus_embeddings)
            top_results = torch.topk(dot_scores, k=self.k)
            hits = [
                res.indices.cpu() if res >= self.DOT_THRESHOLD else None
                for res in top_results
            ]
            return hits

        elif method == "euclidean":
            distances = torch.norm(query_embedding - self.corpus_embeddings, dim=1)
            top_results = torch.topk(
                -distances, k=self.k
            )  # Negative because we want smallest distances
            hits = [
                res.indices.cpu() if res >= self.EUC_THRESHOLD else None
                for res in top_results
            ]
            return hits

        elif method == "hnsw":
            return self._search_hnsw(query_embedding)

        else:
            raise ValueError(f"Unsupported search method: {method}")

    def search(self, query, method="cosine"):
        # Bi-encoder search
        query_embedding = self.bi_encoder.encode(
            query, convert_to_tensor=True, device="cuda"
        )
        top_idx = self._search(query_embedding, method)

        if not top_idx:
            return "i don't know the answer"
        else:
            return self.corpus[top_idx]
