import os
from run_niah import benchmark_retriever

RETRIEVERS = [
    "bm25", 
    "dot", 
    "faiss", 
    "lsh_faiss", 
    "lsh_annoy", 
    "lsh_nmslib"
]

FILE_PATHS = [
    "/home/xinyu/ruler/dataset/synthetic_data_single_qkv_numbers.jsonl",
    "/home/xinyu/ruler/dataset/synthetic_data_single_qkv_uuids.jsonl",
    "/home/xinyu/ruler/dataset/synthetic_data_single_qkv_words.jsonl"
]

top_k_options = [1, 2, 4, 8]
CHUNK_SIZES = [32, 64, 128]
SAMPLES = 40
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GPU = "0"

OUTPUT_DIR = "benchmark_outputs_niah_topk_2_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_output_filename(retriever, file_path, chunk_size, top_k):
    """ Generate output filename based on parameters. """
    dataset_name = os.path.basename(file_path).replace(".jsonl", "")
    return f"{retriever}_{chunk_size}_{dataset_name}_{top_k}.json"


def check_task_exists(output_path):
    """ Check if the output file already exists. """
    return os.path.exists(output_path)


def run_retriever(retriever, file_path, chunk_size, top_k):
    """ Run the benchmark for a specific retriever, file path, and chunk size. """
    output_filename = generate_output_filename(retriever, file_path, chunk_size, top_k)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Skip if the task is already completed
    if check_task_exists(output_path):
        print(f"Skipping {output_filename}: Already exists.")
        return

    print(f"\n--- Running {retriever} | File: {os.path.basename(file_path)} | Chunk Size: {chunk_size} | Top_k: {top_k} ---")

    benchmark_retriever(
        dataset_name="niah",
        file_path=file_path,
        retriever_name=retriever,
        embedding_model=EMBEDDING_MODEL,
        num_samples=SAMPLES,
        gpu=GPU,
        chunk_size=chunk_size,
        output_path=output_path,
        top_k=top_k
    )

    print(f"Results saved to {output_path}")


def main():
    for retriever in RETRIEVERS:
        for file_path in FILE_PATHS:
            for chunk_size in CHUNK_SIZES:
                for top_k in top_k_options:
                    run_retriever(retriever, file_path, chunk_size, top_k)


if __name__ == "__main__":
    main()