import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from .base import BaseRetriever

class FaissRetriever(BaseRetriever):
    def __init__(self, corpus, model_name='sentence-transformers/all-MiniLM-L6-v2', use_gpu=True):
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        self.corpus = corpus
        self.embeddings = self.model.encode(corpus, convert_to_numpy=True, device=self.device)
        faiss.normalize_L2(self.embeddings)

        self.use_gpu = use_gpu
        dim = self.embeddings.shape[1]
        index_cpu = faiss.IndexFlatIP(dim)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0
        else:
            self.index = index_cpu

        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=10):
        query_vec = self.model.encode([query], convert_to_numpy=True, device=self.device)
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, top_k)
        return [self.corpus[i] for i in I[0]]