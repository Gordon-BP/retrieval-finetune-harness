# @title 4. TRAIN script trains TAS-B using Generative Pseudo Labeling (GPL)
from income.jpq.dataset import TextTokenIdsCache, SequenceDataset, pack_tensor_2D
from income.jpq.models.backbones import DistilBertDot, RobertaDot, BertDot
from income.jpq.models.backbones.roberta_tokenizer import RobertaTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    AutoConfig,
    AutoTokenizer,
)
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
from enum import Enum
import os, sys
import torch
import random
import time
import faiss
import logging
import argparse
import numpy as np
import json
import csv
import datetime
from tqdm import tqdm, trange
from collections import defaultdict

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


class TrainQueryDataset(SequenceDataset):
    def __init__(self, queryids_cache, rel_file, max_query_length):
        SequenceDataset.__init__(self, queryids_cache, max_query_length)
        self.reldict = defaultdict(list)
        self.negdict = defaultdict(list)
        for line in tqdm(open(rel_file), desc=os.path.split(rel_file)[1]):
            qid, _, pid, _ = line.split()
            qid, pid = int(qid), int(pid)
            self.reldict[qid].append((pid))

    def __getitem__(self, item):
        ret_val = super().__getitem__(item)
        ret_val["rel_ids"] = self.reldict[item]
        ret_val["neg_ids"] = self.negdict[item]
        return ret_val


class ModelBackbone(Enum):
    DISTILBERT = "distilbert"
    ROBERTA = "roberta"
    BERT = "bert"


@dataclass
class GPLTrainingPipeline:
    log_dir: str
    model_save_dir: str
    init_index_path: str
    init_model_path: str
    preprocess_dir: str
    data_path: str
    logging_steps: int = 50
    gpu_search: bool = True
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu: int = 1
    max_seq_length: int = 64
    train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 2
    weight_decay: float = 0.01
    centroid_weight_decay: float = 0.0
    centroid_lr: float = 1e-4
    lr: float = 5e-6
    lambda_cut: int = 200
    adam_epsilon: float = 1e-8
    warmup_steps: int = 2000
    loss_neg_topK: int = 25
    max_grad_norm: float = 1.0
    seed: int = 42
    init_backbone: ModelBackbone = ModelBackbone.DISTILBERT

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def save_model(model, output_dir, save_name, optimizer=None):
        save_dir = os.path.join(output_dir, save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(save_dir)
        torch.save(os.path.join(save_dir, "training_args.bin"))
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))

    @staticmethod
    def get_collate_function(max_seq_length):
        cnt = 0

        def collate_function(batch):
            nonlocal cnt
            length = None
            if cnt < 10:
                length = max_seq_length
                cnt += 1
            keys = [x.keys() for x in batch]
            input_ids = [x["input_ids"] for x in batch]
            attention_mask = [x["attention_mask"] for x in batch]
            data = {
                "input_ids": pack_tensor_2D(
                    input_ids, default=1, dtype=torch.int64, length=length
                ),
                "attention_mask": pack_tensor_2D(
                    attention_mask, default=0, dtype=torch.int64, length=length
                ),
            }
            qids = [x["id"] for x in batch]
            all_rel_pids = [x["rel_ids"] for x in batch]
            all_neg_pids = [x["neg_ids"] for x in batch]
            return data, qids, all_rel_pids, all_neg_pids

        return collate_function

    @staticmethod
    def get_doc_embeds(psg_ids, pq_codes, centroids):
        M = centroids.shape[0]
        first_indices = torch.arange(M).to(psg_ids.device)
        first_indices = first_indices.expand(len(psg_ids), M).reshape(-1)
        second_indices = pq_codes[psg_ids].reshape(-1)
        embeddings = centroids[first_indices, second_indices].reshape(len(psg_ids), -1)
        return embeddings

    def compute_loss(
        self,
        query_embeddings,
        pq_codes,
        centroids,
        batch_neighbors,
        qids,
        all_rel_pids,
        all_neg_ids,
        lambda_cut,
        corpus,
        queries,
        cross_encoder,
    ):
        loss = 0
        mrr = 0
        train_batch_size = len(batch_neighbors)
        loss_function = torch.nn.MSELoss(reduction="mean")
        for qid, qembedding, retrieve_pids, cur_rel_pids, cur_neg_ids in zip(
            qids, query_embeddings, batch_neighbors, all_rel_pids, all_neg_ids
        ):
            # Removes duplicates from cur_rel_pids
            # and converts it to a Long tensor and moves it to the appropriate device (CPU or GPU).
            cur_rel_pids = list(set(cur_rel_pids))
            cur_rel_pids = torch.LongTensor(cur_rel_pids).to(batch_neighbors.device)

            # checks if it matches any of the current relevant passage IDs (cur_rel_pids).
            # It generates a boolean tensor (target_labels) indicating which passages are relevant.
            target_labels = (retrieve_pids[:, None] == cur_rel_pids).any(-1)

            # Filters the retrieved passage IDs to only include the relevant ones based on target_labels
            retrieved_rel_pids = retrieve_pids[target_labels]

            # updates retrieve_pids to include the missed relevant passages
            # and updates the target_labels accordingly.
            if len(retrieved_rel_pids) < len(cur_rel_pids):
                not_retrieved_rel_pids = cur_rel_pids[
                    (cur_rel_pids[:, None] != retrieved_rel_pids).all(-1)
                ]
                assert len(not_retrieved_rel_pids) > 0
                retrieve_pids = torch.hstack([retrieve_pids, not_retrieved_rel_pids])
                target_labels = torch.hstack(
                    [
                        target_labels,
                        torch.tensor([True] * len(not_retrieved_rel_pids)).to(
                            target_labels.device
                        ),
                    ]
                )

            # Retrieves the embeddings for the given passage IDs.
            psg_embeddings = self.get_doc_embeds(retrieve_pids, pq_codes, centroids)

            # Calculates the scores for the retrieved passages
            # by performing a dot product with the query embedding.
            cur_top_scores = (qembedding.reshape(1, -1) * psg_embeddings).sum(-1)

            # Divides the scores into two categories
            rel_scores = cur_top_scores[target_labels]
            irrel_scores = cur_top_scores[~target_labels]
            print(
                f"Rel scores are of type {type(rel_scores)} and are of size {len(rel_scores)}"
            )
            print(rel_scores)
            print(
                f"Irrel scores are of type {type(irrel_scores)} and are of size {len(irrel_scores)}"
            )
            print(irrel_scores)

            # Old code below was giving error because rel_scores was of len 9 and irrel_scores of len 24
            # Now we truncate irrel_scores of the size of rel_scores
            if len(irrel_scores) > len(rel_scores):
                irrel_scores = irrel_scores[: len(rel_scores)]
            pair_diff = rel_scores - irrel_scores
            # This probably has an impact downstream, and we could consider doing something different

            # Uses the cross-encoder to predict relevance scores for each query-passage pair.
            query, docs = queries[qid], [
                corpus[doc_id] for doc_id in retrieve_pids.tolist()
            ]
            scores_ce = cross_encoder.predict(
                [(query, x) for x in docs],
                show_progress_bar=False,
                convert_to_numpy=False,
                convert_to_tensor=True,
                batch_size=128,
            )

            # cross-encoder scores are divided into relevant and irrelevant
            rel_ce_scores = scores_ce[target_labels]
            irrel_ce_scores = scores_ce[~target_labels]
            # Now we truncate irrel_scores of the size of rel_scores
            if len(irrel_ce_scores) > len(rel_ce_scores):
                irrel_ce_scores = irrel_ce_scores[: len(rel_ce_scores)]

            labels = rel_ce_scores - irrel_ce_scores
            # The MSE loss is calculated between the pair_diff and labels,
            # and added to the cumulative loss.
            cur_loss = loss_function(pair_diff, labels)
            loss += cur_loss

        # Divides the MRR and the cumulative loss by the batch size to average it out.
        mrr /= train_batch_size
        loss /= train_batch_size
        return loss, mrr

    def train(
        self, model, pq_codes, centroid_embeds, opq_transform, opq_index, tokenizer
    ):
        """Train the model"""
        ivf_index = faiss.downcast_index(opq_index.index)
        if self.gpu_search:
            res = faiss.StandardGpuResources()
            res.setTempMemory(128 * 1024 * 1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = len(pq_codes) >= 56
            gpu_index = faiss.index_cpu_to_gpu(res, 0, opq_index, co)

        tb_writer = SummaryWriter(self.log_dir)

        train_dataset = TrainQueryDataset(
            TextTokenIdsCache(self.preprocess_dir, "train-query"),
            os.path.join(self.preprocess_dir, "train-qrel.tsv"),
            self.max_seq_length,
        )

        train_sampler = RandomSampler(train_dataset)
        collate_fn = self.get_collate_function(self.max_seq_length)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.train_batch_size,
            collate_fn=collate_fn,
        )
        t_total = (
            len(train_dataloader)
            // self.gradient_accumulation_steps
            * self.num_train_epochs
        )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [centroid_embeds],
                "weight_decay": self.centroid_weight_decay,
                "lr": self.centroid_lr,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
        )

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Train!
        print("***** Running training *****")
        print("  Num examples = ", len(train_dataset))
        print("  Num Epochs = ", self.num_train_epochs)
        print(
            "  Total train batch size (w. accumulation) = ",
            self.train_batch_size * self.gradient_accumulation_steps,
        )
        print("  Gradient Accumulation steps = ", self.gradient_accumulation_steps)
        print("  Total optimization steps = ", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        tr_mrr, logging_mrr = 0.0, 0.0
        optimizer.zero_grad()
        train_iterator = trange(int(self.num_train_epochs), desc="Epoch")
        self.set_seed()  # Added here for reproductibility (even between python 2 and 3)
        corpus, queries = {}, {}

        def fix_nulls(s):
            for line in s:
                yield line.replace("\0", "")

        reader = csv.reader(
            fix_nulls(
                open(os.path.join(self.data_path, "collection.tsv"), encoding="utf-8")
            ),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        for id, row in enumerate(reader):
            corpus[int(row[0])] = row[1]

        reader = csv.reader(
            fix_nulls(
                open(os.path.join(self.data_path, "queries.tsv"), encoding="utf-8")
            ),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        for id, row in enumerate(reader):
            queries[int(row[0])] = row[1]

        cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=self.model_device,
            max_length=350,
        )

        for epoch_idx, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, (batch, qids, all_rel_poffsets, all_neg_ids) in enumerate(
                epoch_iterator
            ):
                batch = {k: v.to(self.model_device) for k, v in batch.items()}
                model.train()
                query_embeddings = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )

                if self.gpu_search:
                    batch_neighbors = gpu_index.search(
                        query_embeddings.detach().cpu().numpy(), self.loss_neg_topK
                    )[1]
                else:
                    batch_neighbors = opq_index.search(
                        query_embeddings.detach().cpu().numpy(), self.loss_neg_topK
                    )[1]
                batch_neighbors = torch.tensor(batch_neighbors).to(model.device)

                loss, mrr = self.compute_loss(
                    query_embeddings @ opq_transform.T,
                    pq_codes,
                    centroid_embeds,
                    batch_neighbors,
                    qids,
                    all_rel_poffsets,
                    all_neg_ids,
                    self.lambda_cut,
                    corpus,
                    queries,
                    cross_encoder,
                )
                # tr_mrr += mrr
                loss /= self.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                if self.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel (not distributed) training

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    # model.zero_grad()
                    global_step += 1
                    faiss.copy_array_to_vector(
                        centroid_embeds.detach().cpu().numpy().ravel(),
                        ivf_index.pq.centroids,
                    )
                    if self.gpu_search:
                        gpu_index = None
                        gpu_index = faiss.index_cpu_to_gpu(res, 0, opq_index, co)
                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        cur_loss = (tr_loss - logging_loss) / self.logging_steps
                        tb_writer.add_scalar("train/all_loss", cur_loss, global_step)
                        logging_loss = tr_loss

                if step == 45000:
                    self.save_model(model, self.model_save_dir, f"step-{step}")
                    tokenizer.save_pretrained(
                        os.path.join(self.model_save_dir, f"step-{step}")
                    )
                    model.config.save_pretrained(
                        os.path.join(self.model_save_dir, f"step-{step}")
                    )
                    faiss.write_index(
                        opq_index,
                        os.path.join(
                            self.model_save_dir,
                            f"step-{step}",
                            os.path.basename(self.init_index_path),
                        ),
                    )

            self.save_model(model, self.model_save_dir, f"epoch-{epoch_idx+1}")
            tokenizer.save_pretrained(
                os.path.join(self.model_save_dir, f"epoch-{epoch_idx+1}")
            )
            model.config.save_pretrained(
                os.path.join(self.model_save_dir, f"epoch-{epoch_idx+1}")
            )
            faiss.write_index(
                opq_index,
                os.path.join(
                    self.model_save_dir,
                    f"epoch-{epoch_idx+1}",
                    os.path.basename(self.init_index_path),
                ),
            )

    def train_gpl(self):
        logger.warning("Model Device: %s, n_gpu: %s", self.model_device, self.n_gpu)

        if self.init_backbone == ModelBackbone.DISTILBERT:
            print("loading Model for GPL training: {}".format(self.init_model_path))
            config = AutoConfig.from_pretrained(self.init_model_path)
            config.return_dict = False
            config.gradient_checkpointing = self.gpu_search  # to save cuda memory
            model = DistilBertDot.from_pretrained(self.init_model_path, config=config)
            tokenizer = AutoTokenizer.from_pretrained(self.init_model_path)

        elif self.init_backbone == ModelBackbone.BERT:
            print("loading Model for GenQ training: {}".format(self.init_model_path))
            config = AutoConfig.from_pretrained(self.init_model_path)
            config.return_dict = False
            config.gradient_checkpointing = self.gpu_search  # to save cuda memory
            model = BertDot.from_pretrained(self.init_model_path, config=config)
            tokenizer = AutoTokenizer.from_pretrained(self.init_model_path)

        elif self.init_backbone == ModelBackbone.ROBERTA:
            print("loading Model for GPL training: {}".format(self.init_model_path))
            config = RobertaConfig.from_pretrained(self.init_model_path)
            config.return_dict = False
            config.gradient_checkpointing = self.gpu_search  # to save cuda memory
            model = RobertaDot.from_pretrained(self.init_model_path, config=config)
            tokenizer = RobertaTokenizer.from_pretrained(
                "roberta-base", do_lower_case=True
            )

        model.to(self.model_device)

        # Reads a FAISS index
        opq_index = faiss.read_index(self.init_index_path)

        # accessing the first element in the transformation chain of the opq_index.
        vt = faiss.downcast_VectorTransform(opq_index.chain.at(0))
        assert isinstance(vt, faiss.LinearTransform)

        # Extracts the transformation matrix A from the linear transform and reshapes it.
        # This matrix contains the weights used for the linear transformation.
        opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
        # Converts the OPQ transformation matrix to a PyTorch tensor and moves it to a specified device
        opq_transform = torch.FloatTensor(opq_transform).to(self.model_device)

        # Fetches the IVF (Inverted File system) index from the OPQ index.
        ivf_index = faiss.downcast_index(opq_index.index)

        # Extracts the inverted lists from the IVF index.
        # Inverted lists store the ids of vectors assigned to each or cluster in the IVF quantizer
        invlists = faiss.extract_index_ivf(ivf_index).invlists

        # This line retrieves the PQ (Product Quantization) codes of the vectors in the first inverted list.
        # These codes are compact representations of the original vectors.
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)

        # Reshapes the PQ codes and then converts them to a PyTorch tensor.
        # The tensor is then moved to the desired device.
        pq_codes = pq_codes.reshape(-1, invlists.code_size)
        pq_codes = torch.LongTensor(pq_codes).to(self.model_device)

        # Retrieves the centroids of the PQ codebook and reshapes it
        centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
        centroid_embeds = centroid_embeds.reshape(
            ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub
        )
        print(
            f"Centroid Embeds is of type {type(centroid_embeds)} and is {centroid_embeds.size} large"
        )

        # Fetches the coarse quantizer from the IVF index.
        # The coarse quantizer is responsible for partitioning the dataset into clusters
        coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        print(
            f"Coarse Quantizer if of type {type(coarse_quantizer)} and is {coarse_quantizer.ntotal} large"
        )

        # Attempts to retrieve the centroids of the coarse quantizer
        coarse_embeds = faiss.rev_swig_ptr(
            coarse_quantizer.get_xb(), coarse_quantizer.ntotal * coarse_quantizer.d
        )
        coarse_embeds = coarse_embeds.reshape(
            coarse_quantizer.ntotal, coarse_quantizer.d
        )

        # Adds the coarse centroids (from coarse_quantizer) to the PQ centroids
        centroid_embeds += coarse_embeds.reshape(ivf_index.pq.M, -1, ivf_index.pq.dsub)
        print(
            f"Coarse Embeds is of type {type(coarse_embeds)} and is {coarse_embeds.size} large"
        )

        # Updates the centroids in the PQ codebook with the modified values.
        faiss.copy_array_to_vector(centroid_embeds.ravel(), ivf_index.pq.centroids)

        # Zeroes out the centroids of the coarse quantizer.
        coarse_embeds[:] = 0

        # Converts the centroids to a PyTorch tensor and moves them to the desired device.
        coarse_quantizer.add(coarse_embeds)

        # Test to see if we did anything
        # coarse_embeds2 = faiss.rev_swig_ptr(coarse_quantizer.get_xb(), coarse_quantizer.ntotal*coarse_quantizer.d)
        # coarse_embeds2 = coarse_embeds2.reshape(coarse_quantizer.ntotal, coarse_quantizer.d)
        # print(f"Coarse Embeds 2 is of type {type(coarse_embeds2)} and is {coarse_embeds2.size} large" )
        # print(f"Here are the first 10 values of coarse_embeds2:\n{coarse_embeds2[0][:10]}")
        # OLD CODE: faiss.copy_array_to_vector(coarse_embeds.ravel(), coarse_quantizer.pq.centroids)
        centroid_embeds = torch.FloatTensor(centroid_embeds).to(self.model_device)

        # Now the tensor can be involved in gradient-based operations
        centroid_embeds.requires_grad = True

        self.train(
            model, pq_codes, centroid_embeds, opq_transform, opq_index, tokenizer
        )


jpq_pipeline = GPLTrainingPipeline(
    log_dir="./logs",
    model_save_dir="./final_models/nfcorpus/gpl",
    init_index_path="./init/nfcorpus/OPQ96,IVF1,PQ96x8.index",
    init_model_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    preprocess_dir="./preprocessed/nfcorpus",
    data_path="./datasets/nfcorpus",
)
jpq_pipeline.train_gpl()
