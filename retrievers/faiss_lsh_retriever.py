# -------- File: faiss_lsh_retriever.py --------
import time
import os
import faiss
import numpy as np
import torch
from .base import BaseRetriever
from embedding_models.hf_transformer import HFTransformerEmbedding

class FaissLSHRetriever(BaseRetriever):
    def __init__(self, model_name=None, use_gpu=True, hash_bits=128):
        super().__init__(model_name, use_gpu)
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = HFTransformerEmbedding(model_name=model_name, device=self.device)
        self.hash_bits = hash_bits

    def build_index(self, context_chunks):
        t0 = time.time()
        self.embeddings = self.encoder.encode(context_chunks, convert_to_numpy=True, normalize=True)
        self._embed_time = time.time() - t0

        t1 = time.time()
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexLSH(dim, self.hash_bits)
        self.index.add(self.embeddings.astype(np.float32))
        self.context_chunks = context_chunks
        self._index_time = time.time() - t1

    def retrieve(self, query, context_chunks, top_k=10):
        t0 = time.perf_counter()
        query_vec = self.encoder.encode([query], convert_to_numpy=True, normalize=True)
        self._query_embed_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        D, I = self.index.search(query_vec.astype(np.float32), top_k)
        self._search_time = time.perf_counter() - t1
        print(self._search_time)
        return [self.context_chunks[i] for i in I[0] if i != -1]