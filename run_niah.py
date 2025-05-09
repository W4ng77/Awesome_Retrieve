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
    Compute accuracy by checking if any retrieved chunk contains the answer.
    """
    for chunk in retrieved_chunks:
        if reference_answer.lower() in chunk.lower():
            return 1.0
    return 0.0

def benchmark_retriever(dataset_name, file_path, retriever_name, embedding_model, num_samples=1000, gpu=None, chunk_size=16):
    """
    Benchmark the retriever on the dataset.
    
    Args:
        gpu (str or None): GPU ID (e.g., "0" or "1"). If None, uses CPU.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Determine device
    if gpu is not None:
        available_gpus = torch.cuda.device_count()
        try:
            gpu = int(gpu)
            if gpu >= available_gpus:
                raise ValueError(f"Invalid GPU ID: {gpu}. Available GPUs: {list(range(available_gpus))}")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            device = f"cuda:0"
            print(f"[INFO] Using GPU {gpu}")
        except ValueError as e:
            print(f"[ERROR] {e}. Falling back to CPU.")
            device = "cpu"
    else:
        device = "cpu"
        print("[INFO] Using CPU")

    os.environ["EMBEDDING_MODEL"] = embedding_model
    data, chunking_info = load_dataset(dataset_name, file_path, num_samples=num_samples, chunk_size=chunk_size, embedding_model=embedding_model)

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    retriever_args = {
        "model_name": embedding_model,
        "use_gpu": gpu is not None,
        # "device": device
    }

    retriever = RETRIEVER_CLASSES[retriever_name](**retriever_args)

    overall_results = []
    accuracy_scores = []

    print(f"\n--- Running {retriever_name} on {device} with model {embedding_model} ---")

    for sample_idx, sample in enumerate(tqdm(data)):
        query = sample["query"]
        context_chunks = sample["context"]
        reference_answer = sample["answer"]

        # Timing metrics
        sample_embed_time = 0.0
        sample_index_time = 0.0
        sample_query_embed_time = 0.0
        sample_search_time = 0.0

        # Build index (context_chunks is just one item in the list)
        if hasattr(retriever, "build_index"):
            retriever.build_index(context_chunks)
            sample_embed_time += getattr(retriever, "_embed_time", 0.0)
            sample_index_time += getattr(retriever, "_index_time", 0.0)

        # Retrieve
        retrieved = retriever.retrieve(query=query, top_k=1)
        sample_query_embed_time += getattr(retriever, "_query_embed_time", 0.0)
        sample_search_time += getattr(retriever, "_search_time", 0.0)

        # Calculate accuracy
        accuracy = compute_accuracy(retrieved, reference_answer)
        accuracy_scores.append(accuracy)

        sample_result = {
            "sample_idx": sample_idx,
            "accuracy": round(accuracy, 4),
            "embedding_time": round(sample_embed_time, 5),
            "indexing_time": round(sample_index_time, 5),
            "query_embed_time": round(sample_query_embed_time, 5),
            "search_time": round(sample_search_time, 5)
        }

        overall_results.append(sample_result)

    # Average accuracy
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0

    # Save results to JSON
    output_dir = "benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)

    gpu_suffix = f"gpu{gpu}" if gpu is not None else "cpu"
    output_path = os.path.join(output_dir, f"{retriever_name}_{dataset_name}_accuracy_{gpu_suffix}.json")

    with open(output_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\nBenchmark completed. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="niah")
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--retriever", type=str, choices=list(RETRIEVER_CLASSES.keys()), required=True)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default=None, help="GPU ID to use (e.g., 0, 1). Leave empty for CPU.")
    parser.add_argument("--chunk_size", type=int, default=16)
    args = parser.parse_args()

    benchmark_retriever(
        dataset_name=args.dataset,
        file_path=args.file_path,
        retriever_name=args.retriever,
        embedding_model=args.embedding_model,
        num_samples=args.samples,
        gpu=args.gpu,
        chunk_size=args.chunk_size
    )
