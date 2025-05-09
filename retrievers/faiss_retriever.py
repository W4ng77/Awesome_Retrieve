import time
import numpy as np
import torch
import os
from .base import BaseRetriever
from embedding_models.hf_transformer import HFTransformerEmbedding
from sentence_transformers import SentenceTransformer
import faiss
class FaissRetriever(BaseRetriever):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', use_gpu=True):
        super().__init__(model_name, use_gpu)
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.use_gpu = use_gpu

    def build_index(self, context_chunks):
        t0 = time.time()
        embeddings = self.model.encode(context_chunks, convert_to_numpy=True, device=self.device)
        self._embed_time = time.time() - t0

        t1 = time.time()
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(embeddings)
        self.index = index
        self.context_chunks = context_chunks
        self._index_time = time.time() - t1

    def retrieve(self, query, top_k=10):
        t0 = time.time()
        query_vec = self.model.encode([query], convert_to_numpy=True, device=self.device)
        faiss.normalize_L2(query_vec)
        self._query_embed_time = time.time() - t0

        t1 = time.time()
        D, I = self.index.search(query_vec, top_k)
        self._search_time = time.time() - t1

        return [self.context_chunks[i] for i in I[0]]
