import argparse
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset.load_babilong_qa1_128k import load_babilong_qa1
from retrievers import RETRIEVER_CLASSES

def benchmark_retriever(dataset_name, retriever_name, embedding_model, num_samples=200, use_gpu=True, chunk_size=16, length_splits=None):
    os.environ["EMBEDDING_MODEL"] = embedding_model

    if length_splits is None:
        length_splits = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M"]

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    retriever_args = {"model_name": embedding_model, "use_gpu": use_gpu}
    retriever = RETRIEVER_CLASSES[retriever_name](**retriever_args)

    overall_results = []

    output_dir = "benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{retriever_name}_{dataset_name}_chunk{chunk_size}_results.json")

    # Load existing data to avoid overwriting
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    for length_split in length_splits:
        print(f"\n--- Running {retriever_name} on {dataset_name} - {length_split} split | chunk_size={chunk_size} ---")

        data, chunking_info = load_babilong_qa1(
            length_split=length_split,
            num_samples=num_samples,
            chunk_size=chunk_size,
            embedding_model=embedding_model
        )

        hits = 0
        total_queries = 0

        for sample_idx, sample in enumerate(tqdm(data)):
            queries = sample["queries"]
            context_chunks = sample["context"]
            reference_answers = sample["answers"]

            # Initialize timers
            sample_embed_time = 0.0
            sample_index_time = 0.0
            sample_query_embed_time = 0.0
            sample_search_time = 0.0

            # Build index
            if hasattr(retriever, "build_index"):
                # Ensure there is content to index
                if context_chunks:
                    retriever.build_index(context_chunks)
                    sample_embed_time += getattr(retriever, "_embed_time", 0.0)
                    sample_index_time += getattr(retriever, "_index_time", 0.0)
                else:
                    print(f"Warning: Empty context chunks for {length_split}. Skipping indexing.")

            # Process each query
            for query, ref_answer in zip(queries, reference_answers):
                retrieved = retriever.retrieve(query=query, context_chunks=context_chunks, top_k=10)
                sample_query_embed_time += getattr(retriever, "_query_embed_time", 0.0)
                sample_search_time += getattr(retriever, "_search_time", 0.0)

                # Check if target is contained in any retrieved chunk
                target = ref_answer[0].lower()
                hit = any(target in chunk.lower() for chunk in retrieved)
                hits += int(hit)
                total_queries += 1

        # Calculate accuracy
        accuracy = hits / total_queries if total_queries else 0.0

        # Store result for this split
        split_result = {
            "length_split": length_split,
            "accuracy": round(accuracy, 4),
            "total_queries": total_queries,
            "hits": hits,
            "embedding_time": round(sample_embed_time, 5),
            "indexing_time": round(sample_index_time, 5),
            "query_embed_time": round(sample_query_embed_time, 5),
            "search_time": round(sample_search_time, 5)
        }

        # Append new results for each split
        existing_data.append(split_result)
        existing_data.append(chunking_info)

    # Write all results back to the file
    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print(f"\nBenchmark completed. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="babilong_qa1")
    parser.add_argument("--retriever", type=str, choices=list(RETRIEVER_CLASSES.keys()), required=True)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--length_splits", type=str, nargs="+", default=["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M"])

    args = parser.parse_args()

    benchmark_retriever(
        dataset_name=args.dataset,
        retriever_name=args.retriever,
        embedding_model=args.embedding_model,
        num_samples=args.samples,
        use_gpu=args.gpu,
        chunk_size=args.chunk_size,
        length_splits=args.length_splits
    )
