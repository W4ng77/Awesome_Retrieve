from sentence_transformers import SentenceTransformer
import torch
from .base import EmbeddingModel

class HFTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda:0"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, convert_to_numpy=True, normalize=True):
        return self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize,
            device=self.model.device
        )