# -------- Updated Retriever Implementations with Timing --------

# 1. DotProductRetriever
import time
import numpy as np
import torch
import os
from .base import BaseRetriever
from embedding_models.hf_transformer import HFTransformerEmbedding

class DotProductRetriever(BaseRetriever):
    def __init__(self, model_name=None, use_gpu=True):
        super().__init__(model_name, use_gpu)
        self.device = "cuda:7" if use_gpu and torch.cuda.is_available() else "cpu"
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = HFTransformerEmbedding(model_name=model_name, device=self.device)

    def build_index(self, context_chunks):
        t0 = time.time()
        self.embeddings = self.encoder.encode(context_chunks, convert_to_numpy=True, normalize=True)
        self._embed_time = time.time() - t0
        self.context_chunks = context_chunks
        self._index_time = 0

    def retrieve(self, query, context_chunks, top_k=10):
        t0 = time.time()
        query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize=True)
        self._query_embed_time = time.time() - t0

        t1 = time.time()
        scores = np.dot(self.embeddings, query_embedding.T).squeeze()
        top_indices = np.argsort(-scores)[:top_k]
        self._search_time = time.time() - t1

        return [self.context_chunks[i] for i in top_indices]