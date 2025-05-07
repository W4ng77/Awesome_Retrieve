import time
import numpy as np
import torch
import os
import scann
from .base import BaseRetriever
from embedding_models.hf_transformer import HFTransformerEmbedding
from sentence_transformers import SentenceTransformer

class ScaNNRetriever(BaseRetriever):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', use_gpu=True):
        super().__init__(model_name, use_gpu)
        self.device = "cuda:2" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.use_gpu = use_gpu

    def build_index(self, context_chunks):
        t0 = time.time()
        self.embeddings = self.model.encode(context_chunks, convert_to_numpy=True, device=self.device)
        self._embed_time = time.time() - t0

        t1 = time.time()
        # normalize for dot-product search
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index = (
            scann.scann_ops_pybind.builder(self.embeddings, 10, "dot_product")
            .score_brute_force()  # 👈 明确指定评分机制
            .build()
        )

        self.context_chunks = context_chunks
        self._index_time = time.time() - t1

    def retrieve(self, query, context_chunks=None, top_k=10):
        t0 = time.time()
        query_vec = self.model.encode([query], convert_to_numpy=True, device=self.device)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        self._query_embed_time = time.time() - t0

        t1 = time.time()
        neighbors, _ = self.index.search_batched(query_vec, final_num_neighbors=top_k)
        self._search_time = time.time() - t1

        return [self.context_chunks[i] for i in neighbors[0]]
