from rank_bm25 import BM25Okapi
from .base import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=10):
        scores = self.bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self.corpus[i] for i in top_indices]
