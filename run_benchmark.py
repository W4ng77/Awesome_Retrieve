# -------- File: run_benchmark.py --------
import argparse
import csv
import time
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset.load_hotpotqa import load_hotpotqa_subset
from dataset.build_hotpotqa_128k_sample import load_hotpotqa_concatenated
from retrievers import RETRIEVER_CLASSES
import json

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


# def load_dataset(name, num_samples=200, chunk_size=16, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
#     if name == "hotpotqa":
#         ds = load_hotpotqa_subset(num_samples=num_samples, embedding_model=embedding_model, chunk_size=chunk_size)
#         for item in ds:
#             item["gold"] = item.get("answers", [""])[0]
#         return ds
#     else:
#         raise ValueError(f"Unsupported dataset: {name}")

def load_dataset(name, num_samples=200, chunk_size=16, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    if name == "hotpotqa":
        ds, chunking_info = load_hotpotqa_concatenated(split="test", num_samples=num_samples, embedding_model=embedding_model, chunk_size=chunk_size)
        return ds, chunking_info
    else:
        raise ValueError(f"Unsupported dataset: {name}")


# def benchmark_retriever(dataset_name, retriever_name, embedding_model, num_samples=200, use_gpu=True, chunk_size=16):
#     os.environ["EMBEDDING_MODEL"] = embedding_model
#     data = load_dataset(dataset_name, num_samples=num_samples, chunk_size=chunk_size, embedding_model=embedding_model)

#     tokenizer = AutoTokenizer.from_pretrained(embedding_model)

#     retriever_args = {"model_name": embedding_model, "use_gpu": use_gpu}
#     retriever = RETRIEVER_CLASSES[retriever_name](**retriever_args)

#     f1_scores = []
#     embed_time = 0.0
#     index_time = 0.0
#     query_embed_time = 0.0
#     search_time = 0.0

#     print(f"\n--- Running {retriever_name} with model {embedding_model} on {dataset_name} | chunk_size={chunk_size} ---")
#     for sample in tqdm(data):
#         query = sample["query"]
#         context_chunks = sample["context"]
#         reference_answer = sample["gold"]

#         if hasattr(retriever, "build_index"):
#             retriever.build_index(context_chunks)
#             embed_time += getattr(retriever, "_embed_time", 0.0)
#             index_time += getattr(retriever, "_index_time", 0.0)

#         retrieved = retriever.retrieve(query=query, context_chunks=context_chunks, top_k=10)
#         query_embed_time += getattr(retriever, "_query_embed_time", 0.0)
#         search_time += getattr(retriever, "_search_time", 0.0)

#         if reference_answer:
#             f1_scores.append(compute_best_f1(retrieved, reference_answer))

#     avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

#     print(f"\nAvg F1@Top10: {avg_f1:.2f}")
#     print(f"Embedding time: {embed_time:.2f}s | Indexing time: {index_time:.2f}s | Query Embedding: {query_embed_time:.2f}s | Search: {search_time:.2f}s")

#     # 保存结果到 JSON 文件
#     result_file = "benchmark_results.json"
#     result_data = {
#         "dataset": dataset_name,
#         "retriever": retriever_name,
#         "embedding_model": embedding_model,
#         "samples": num_samples,
#         "chunk_size": chunk_size,
#         "f1_top10": round(avg_f1, 4),
#         "embedding_time": round(embed_time, 2),
#         "indexing_time": round(index_time, 2),
#         "query_embed_time": round(query_embed_time, 2),
#         "search_time": round(search_time, 2)
#     }

#     # 读取现有数据（如果有）
#     if os.path.exists(result_file):
#         with open(result_file, "r") as f:
#             try:
#                 all_results = json.load(f)
#             except json.JSONDecodeError:
#                 all_results = []
#     else:
#         all_results = []

#     # 添加新结果
#     all_results.append(result_data)

#     # 写回 JSON 文件
#     with open(result_file, "w") as f:
#         json.dump(all_results, f, indent=2)

def benchmark_retriever(dataset_name, retriever_name, embedding_model, num_samples=200, use_gpu=True, chunk_size=16):
    os.environ["EMBEDDING_MODEL"] = embedding_model
    data,chunking_info = load_dataset(dataset_name, num_samples=num_samples, chunk_size=chunk_size, embedding_model=embedding_model)

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    retriever_args = {"model_name": embedding_model, "use_gpu": use_gpu}
    retriever = RETRIEVER_CLASSES[retriever_name](**retriever_args)

    overall_results = []

    print(f"\n--- Running {retriever_name} with model {embedding_model} on {dataset_name} | chunk_size={chunk_size} ---")

    for sample_idx, sample in enumerate(tqdm(data)):
        #import pdb; pdb.set_trace()
        queries = sample["queries"]
        context_chunks = sample["context"]
        reference_answers = sample["answers"]

        # 每个大样本单独计时
        sample_embed_time = 0.0
        sample_index_time = 0.0
        sample_query_embed_time = 0.0
        sample_search_time = 0.0
        sample_f1_scores = []

        if hasattr(retriever, "build_index"):
            retriever.build_index(context_chunks)
            sample_embed_time += getattr(retriever, "_embed_time", 0.0)
            sample_index_time += getattr(retriever, "_index_time", 0.0)

        # 对每个 query 检索
        for query, ref_answer in zip(queries, reference_answers):
            retrieved = retriever.retrieve(query=query, context_chunks=context_chunks, top_k=10)
            sample_query_embed_time += getattr(retriever, "_query_embed_time", 0.0)
            sample_search_time += getattr(retriever, "_search_time", 0.0)

            if ref_answer:
                sample_f1_scores.append(compute_best_f1(retrieved, ref_answer[0]))  # ref_answer是list

        avg_f1 = sum(sample_f1_scores) / len(sample_f1_scores) if sample_f1_scores else 0.0

        # 记录这个拼接样本的结果
        sample_result = {
            "sample_idx": sample_idx,
            "queries": len(queries),
            "context_chunks": len(context_chunks),
            "f1_top10": round(avg_f1, 4),
            "embedding_time": round(sample_embed_time, 5),
            "indexing_time": round(sample_index_time, 5),
            "query_embed_time": round(sample_query_embed_time, 5),
            "search_time": round(sample_search_time, 5)
        }

        overall_results.append(sample_result)
        overall_results.append(chunking_info)

    # 保存到 JSON 文件
    output_dir = "benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{retriever_name}_{dataset_name}_chunk{chunk_size}_results.json")


    with open(output_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\nBenchmark completed. Results saved to {output_path}")

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
        parser.add_argument("--chunk_size", type=int, default=16)
        args = parser.parse_args()

        benchmark_retriever(
            dataset_name=args.dataset,
            retriever_name=args.retriever,
            embedding_model=args.embedding_model,
            num_samples=args.samples,
            use_gpu=args.gpu,
            chunk_size=args.chunk_size
        )
