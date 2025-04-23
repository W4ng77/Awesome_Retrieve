import time
import numpy as np
import torch
from .base import BaseRetriever
from sentence_transformers import SentenceTransformer
import scann


class ScaNNRetriever(BaseRetriever):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', use_gpu=True):
        super().__init__(model_name, use_gpu)
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.use_gpu = use_gpu

    def build_index(self, context_chunks):
        t0 = time.time()
        # Encode context chunks
        self.embeddings = self.model.encode(context_chunks, convert_to_numpy=True, device=self.device)
        self.context_chunks = context_chunks
        self._embed_time = time.time() - t0

        t1 = time.time()
        self.searcher = scann.scann_ops_pybind.builder(self.embeddings, 10, "dot_product") \
                            .tree(num_leaves=200, num_leaves_to_search=100, training_sample_size=250000) \
                            .score_ah(2, anisotropic_quantization_threshold=0.2) \
                            .reorder(100) \
                            .build()
        self._index_time = time.time() - t1

    def retrieve(self, query, context_chunks=None, top_k=10):
        t0 = time.time()
        query_vec = self.model.encode([query], convert_to_numpy=True, device=self.device)
        self._query_embed_time = time.time() - t0

        t1 = time.time()
        neighbors, _ = self.searcher.search_batched(query_vec, final_num_neighbors=top_k)
        self._search_time = time.time() - t1

        return [self.context_chunks[i] for i in neighbors[0]]
