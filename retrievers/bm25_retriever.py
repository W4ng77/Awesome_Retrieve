# 2. BM25Retriever
from rank_bm25 import BM25Okapi
import time
from .base import BaseRetriever
class BM25Retriever(BaseRetriever):
    def __init__(self, model_name=None, use_gpu=True):
        super().__init__(model_name, use_gpu)

    def build_index(self, context_chunks):
        t0 = time.time()
        self.tokenized_corpus = [doc.split() for doc in context_chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.context_chunks = context_chunks
        self._index_time = time.time() - t0
        self._embed_time = 0

    def retrieve(self,query,top_k=10):
        t0 = time.time()
        self._query_embed_time = 0
        scores = self.bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        self._search_time = time.time() - t0
        return [self.context_chunks[i] for i in top_indices]