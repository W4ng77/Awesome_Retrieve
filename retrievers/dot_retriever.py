import torch
import numpy as np
import os
from .base import BaseRetriever
from embedding_models.hf_transformer import HFTransformerEmbedding

class DotProductRetriever(BaseRetriever):
    def __init__(self, corpus, model_name=None, use_gpu=True):
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = HFTransformerEmbedding(model_name=model_name, device=self.device)

        self.corpus = corpus
        self.embeddings = self.encoder.encode(corpus, convert_to_numpy=True, normalize=True)

    def retrieve(self, query, top_k=10):
        query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize=True)
        scores = np.dot(self.embeddings, query_embedding.T).squeeze()
        top_indices = np.argsort(-scores)[:top_k]
        return [self.corpus[i] for i in top_indices]