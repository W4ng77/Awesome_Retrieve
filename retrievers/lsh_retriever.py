# -------- File: retrievers/lsh_retriever.py --------
import numpy as np
import torch
import os
import faiss
from .base import BaseRetriever
from embedding_models.hf_transformer import HFTransformerEmbedding

class LSHRetriever(BaseRetriever):
    def __init__(self, corpus, model_name=None, use_gpu=True):
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = HFTransformerEmbedding(model_name=model_name, device=self.device)

        self.corpus = corpus
        self.embeddings = self.encoder.encode(corpus, convert_to_numpy=True, normalize=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexLSH(dim, 128)  # 128 bits hash
        self.index.add(self.embeddings.astype(np.float32))

    def retrieve(self, query, top_k=10):
        query_vec = self.encoder.encode([query], convert_to_numpy=True, normalize=True)
        D, I = self.index.search(query_vec.astype(np.float32), top_k)
        return [self.corpus[i] for i in I[0] if i != -1]
