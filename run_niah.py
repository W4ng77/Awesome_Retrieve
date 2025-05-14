import argparse
import time
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from retrievers import RETRIEVER_CLASSES
import json
import torch
from dataset.load_niah import load_niah_dataset

def load_dataset(name, file_path, num_samples=200, chunk_size=16, embedding_model="gpt2"):
    if name == "niah":
        ds, chunking_info = load_niah_dataset(file_path, embedding_model=embedding_model, chunk_size=chunk_size, num_samples=num_samples)
        return ds, chunking_info
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def compute_accuracy(retrieved_chunks, reference_answer):
    """
    Compute accuracy. If any retrieved chunk contains the reference answer, it is considered a hit.
    """
    for chunk in retrieved_chunks:
        if reference_answer.lower() in chunk.lower():
            return 1.0
    return 0.0

def benchmark_retriever(dataset_name, file_path, retriever_name, embedding_model, num_samples, gpu, chunk_size, output_path, top_k):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    device = f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu"

    # Dataset loading
    data, _ = load_dataset(dataset_name, file_path, num_samples=num_samples, chunk_size=chunk_size, embedding_model=embedding_model)

    # Initialize retriever
    retriever_args = {"model_name": embedding_model, "use_gpu": gpu is not None}
    retriever = RETRIEVER_CLASSES[retriever_name](**retriever_args)

    overall_results = []
    
    # Metrics accumulators for averaging
    total_accuracy = 0.0
    total_embed_time = 0.0
    total_index_time = 0.0
    total_query_embed_time = 0.0
    total_search_time = 0.0

    print(f"\n--- Running {retriever_name} on {dataset_name} with {embedding_model} | chunk_size={chunk_size} ---")

    for sample_idx, sample in enumerate(tqdm(data, desc="Benchmarking Samples")):
        query = sample["query"]
        context_chunks = sample["context"]
        reference_answer = sample["answer"]

        # Timing variables for each sample
        embed_time = 0.0
        index_time = 0.0
        query_embed_time = 0.0
        search_time = 0.0

        # Building index
        if hasattr(retriever, "build_index"):
            # 执行 build_index 方法，该方法内部已将 embed_time 和 index_time 分离
            retriever.build_index(context_chunks)

            # 直接提取嵌入时间和索引时间
            embed_time = getattr(retriever, "_embed_time", 0.0)
            index_time = getattr(retriever, "_index_time", 0.0)

        # Single Retrieval Process
        # retrieve 方法内部已将 query_embed_time 和 search_time 分离
        retrieved = retriever.retrieve(query=query, top_k=top_k)

        # 直接提取 query embedding 和 search 时间
        query_embed_time = getattr(retriever, "_query_embed_time", 0.0)
        search_time = getattr(retriever, "_search_time", 0.0)
        # Compute accuracy
        accuracy = compute_accuracy(retrieved, reference_answer)

        # Accumulate metrics
        total_accuracy += accuracy
        total_embed_time += embed_time
        total_index_time += index_time
        total_query_embed_time += query_embed_time
        total_search_time += search_time

        # Record per-sample results
        sample_result = {
            "sample_idx": sample_idx,
            "query": query,
            "accuracy": accuracy,
            "embedding_time": round(embed_time, 5),
            "indexing_time": round(index_time, 5),
            "query_embed_time": round(query_embed_time, 5),
            "search_time": round(search_time, 5),
            "retrieved": retrieved[:top_k]  # Only keep top_k results
        }
        overall_results.append(sample_result)

    # Calculate average metrics
    num_samples = len(data)
    avg_accuracy = total_accuracy / num_samples if num_samples > 0 else 0.0
    avg_embed_time = total_embed_time / num_samples if num_samples > 0 else 0.0
    avg_index_time = total_index_time / num_samples if num_samples > 0 else 0.0
    avg_query_embed_time = total_query_embed_time / num_samples if num_samples > 0 else 0.0
    avg_search_time = total_search_time / num_samples if num_samples > 0 else 0.0

    # Add average metrics as a separate entry
    average_result = {
        "sample_idx": "average",
        "accuracy": round(avg_accuracy, 5),
        "embedding_time": round(avg_embed_time, 5),
        "indexing_time": round(avg_index_time, 5),
        "query_embed_time": round(avg_query_embed_time, 5),
        "search_time": round(avg_search_time, 5),
    }
    overall_results.append(average_result)

    # Save results to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="niah")
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--retriever", type=str, choices=list(RETRIEVER_CLASSES.keys()), required=True)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default=None, help="GPU ID to use (e.g., 0, 1). Leave empty for CPU.")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    benchmark_retriever(
        dataset_name=args.dataset,
        file_path=args.file_path,
        retriever_name=args.retriever,
        embedding_model=args.embedding_model,
        num_samples=args.samples,
        gpu=args.gpu,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        output_path=args.output_path
    )
