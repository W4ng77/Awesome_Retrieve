from datasets import load_dataset
from transformers import AutoTokenizer
import os

# Embedding model path configuration
EMBEDDING_MODEL_PATHS = {
    "bm25": None,
    "faiss": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "sbert": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
}

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL_PATHS.get("faiss"))
TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

def chunk_text(text, max_tokens=16):
    tokens = TOKENIZER(text, return_offsets_mapping=True, truncation=False)
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        if not chunk_ids:
            continue
        start = offsets[i][0]
        end = offsets[min(i + max_tokens - 1, len(offsets)-1)][1]
        chunk = text[start:end]
        chunks.append(chunk.strip())
    return chunks

def load_hotpotqa_subset(split="test", num_samples=200):
    ds = load_dataset("THUDM/LongBench", "hotpotqa", split=split)
    ds = ds.select(range(min(num_samples, len(ds))))
    processed_data = []
    for example in ds:
        query = example["input"]
        context = example["context"]
        context_chunks = chunk_text(context, max_tokens=16)
        processed_data.append({"query": query, "context": context_chunks})
    return processed_data