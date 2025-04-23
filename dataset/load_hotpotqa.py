# -------- File: load_hotpotqa.py --------
from datasets import load_dataset
from transformers import AutoTokenizer

def chunk_text(text, tokenizer, max_tokens=16):
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=False)
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        if i >= len(offsets):
            break
        start = offsets[i][0]
        end = offsets[min(i + max_tokens - 1, len(offsets) - 1)][1]
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def load_hotpotqa_subset(split="test", num_samples=200, embedding_model="sentence-transformers/all-MiniLM-L6-v2", chunk_size=16, return_gold=True):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    ds = load_dataset("THUDM/LongBench", "hotpotqa", split=split)
    ds = ds.select(range(min(num_samples, len(ds))))
    processed_data = []
    for example in ds:
        query = example["input"]
        context = example["context"]
        context_chunks = chunk_text(context, tokenizer, max_tokens=chunk_size)
        entry = {"query": query, "context": context_chunks}
        if return_gold and "answers" in example:
            entry["answers"] = example["answers"]
        processed_data.append(entry)
    return processed_data
