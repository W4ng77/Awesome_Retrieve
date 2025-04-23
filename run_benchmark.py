# -------- File: run_benchmark.py --------
import argparse
import csv
import time
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset.load_hotpotqa import load_hotpotqa_subset
from retrievers import RETRIEVER_CLASSES


def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_best_f1(predictions, reference):
    scores = [compute_f1(pred, reference) for pred in predictions]
    return max(scores) if scores else 0.0


def load_dataset(name, num_samples=200):
    if name == "hotpotqa":
        ds = load_hotpotqa_subset(num_samples=num_samples)
        for item in ds:
            item["gold"] = item.get("answers", [""])[0]
        return ds
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def benchmark_retriever(dataset_name, retriever_name, embedding_model, num_samples=200, use_gpu=True):
    os.environ["EMBEDDING_MODEL"] = embedding_model
    data = load_dataset(dataset_name, num_samples=num_samples)

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    retriever_args = {"model_name": embedding_model, "use_gpu": use_gpu}
    retriever = RETRIEVER_CLASSES[retriever_name](**retriever_args)

    f1_scores = []
    embed_time = 0.0
    index_time = 0.0
    query_embed_time = 0.0
    search_time = 0.0

    print(f"\n--- Running {retriever_name} with model {embedding_model} on {dataset_name} ---")
    for sample in tqdm(data):
        query = sample["query"]
        context_chunks = sample["context"]
        reference_answer = sample["gold"]

        if hasattr(retriever, "build_index"):
            retriever.build_index(context_chunks)
            embed_time += getattr(retriever, "_embed_time", 0.0)
            index_time += getattr(retriever, "_index_time", 0.0)

        retrieved = retriever.retrieve(query=query, context_chunks=context_chunks, top_k=10)
        query_embed_time += getattr(retriever, "_query_embed_time", 0.0)
        search_time += getattr(retriever, "_search_time", 0.0)

        if reference_answer:
            f1_scores.append(compute_best_f1(retrieved, reference_answer))

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print(f"\nAvg F1@Top10: {avg_f1:.2f}")
    print(f"Embedding time: {embed_time:.2f}s | Indexing time: {index_time:.2f}s | Query Embedding: {query_embed_time:.2f}s | Search: {search_time:.2f}s")

    result_file = "benchmark_results.csv"
    write_header = not os.path.exists(result_file)
    with open(result_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "dataset", "retriever", "embedding_model", "samples", "f1_top10",
                "embedding_time", "indexing_time", "query_embed_time", "search_time"
            ])
        writer.writerow([
            dataset_name, retriever_name, embedding_model, num_samples, round(avg_f1, 4),
            round(embed_time, 2), round(index_time, 2), round(query_embed_time, 2), round(search_time, 2)
        ])


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
