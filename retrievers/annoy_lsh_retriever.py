# -------- File: annoy_lsh_retriever.py --------
import time
import os
import numpy as np
import torch
from annoy import AnnoyIndex
from .base import BaseRetriever
from embedding_models.hf_transformer import HFTransformerEmbedding

class AnnoyLSHRetriever(BaseRetriever):
    def __init__(self, model_name=None, use_gpu=True, num_trees=10):
        super().__init__(model_name, use_gpu)
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = HFTransformerEmbedding(model_name=model_name, device=self.device)
        self.num_trees = num_trees

    def build_index(self, context_chunks):
        t0 = time.perf_counter()
        self.embeddings = self.encoder.encode(context_chunks, convert_to_numpy=True, normalize=True)
        self._embed_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        dim = self.embeddings.shape[1]
        self.index = AnnoyIndex(dim, metric='angular')
        for i, vec in enumerate(self.embeddings):
            self.index.add_item(i, vec.tolist())
        self.index.build(self.num_trees)
        self.context_chunks = context_chunks
        self._index_time = time.perf_counter() - t1

    def retrieve(self, query, top_k=10):
        t0 = time.perf_counter()
        query_vec = self.encoder.encode([query], convert_to_numpy=True, normalize=True)
        self._query_embed_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        indices = self.index.get_nns_by_vector(query_vec[0].tolist(), top_k)
        self._search_time = time.perf_counter() - t1

        return [self.context_chunks[i] for i in indices]