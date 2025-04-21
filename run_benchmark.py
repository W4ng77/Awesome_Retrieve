# -------- File: run_benchmark.py --------
import argparse
import time
from tqdm import tqdm
from dataset.load_hotpotqa import load_hotpotqa_subset
from retrievers.bm25_retriever import BM25Retriever
from retrievers.faiss_retriever import FaissRetriever
from retrievers.dot_retriever import DotProductRetriever
from retrievers.lsh_retriever import LSHRetriever
import os
import json
from transformers import AutoTokenizer

RETRIEVER_CLASSES = {
    "bm25": BM25Retriever,
    "faiss": FaissRetriever,
    "dot": DotProductRetriever,
    "lsh": LSHRetriever,
}

def load_dataset(name, num_samples=200):
    if name == "hotpotqa":
        return load_hotpotqa_subset(num_samples=num_samples)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def benchmark_retriever(dataset_name, retriever_name, embedding_model, num_samples=200, use_gpu=True):
    os.environ["EMBEDDING_MODEL"] = embedding_model
    data = load_dataset(dataset_name, num_samples=num_samples)

    # debug: check query + chunk token length
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    for i, sample in enumerate(data):
        query_len = len(tokenizer(sample["query"])['input_ids'])
        max_chunk_len = max(len(tokenizer(c)['input_ids']) for c in sample["context"])
        if query_len > 512 or max_chunk_len > 512:
            print(f"\n⚠️ Sample #{i} too long | query: {query_len} tokens, max chunk: {max_chunk_len} tokens")
            print("Query:", sample["query"][:200], "...\n")

    RetrieverClass = RETRIEVER_CLASSES.get(retriever_name)
    if RetrieverClass is None:
        raise ValueError(f"Unknown retriever method: {retriever_name}")

    total_time = 0
    print(f"\n--- Running {retriever_name} with model {embedding_model} on {dataset_name} ---")
    for sample in tqdm(data):
        retriever = RetrieverClass(sample["context"]) if retriever_name == "bm25" else RetrieverClass(
            sample["context"], model_name=embedding_model, use_gpu=use_gpu
        )
        start = time.time()
        retriever.retrieve(sample["query"], top_k=10)
        total_time += time.time() - start

    print(f"Total time: {total_time:.2f}s | Avg per query: {total_time / len(data):.4f}s")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Use run_all.py for batch benchmark or pass --retriever ...")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="hotpotqa")
        parser.add_argument("--retriever", type=str, choices=list(RETRIEVER_CLASSES.keys()), required=True)
        parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
        parser.add_argument("--samples", type=int, default=200)
        parser.add_argument("--gpu", action="store_true")
        args = parser.parse_args()

        benchmark_retriever(
            dataset_name=args.dataset,
            retriever_name=args.retriever,
            embedding_model=args.embedding_model,
            num_samples=args.samples,
            use_gpu=args.gpu
        )
