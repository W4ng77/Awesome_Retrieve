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
    "/home/linrui/RULER/scripts/data/synthetic/data/single_qkv_numbers/single_qkv_numbers/validation.jsonl",
    "/home/linrui/RULER/scripts/data/synthetic/data/single_qkv_uuids/single_qkv_uuids/validation.jsonl",
    "/home/linrui/RULER/scripts/data/synthetic/data/single_qkv_words/single_qkv_words/validation.jsonl"
]

CHUNK_SIZES = [16, 32, 64, 128]
SAMPLES = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GPU = "3"  # Adjust GPU ID as needed


def run_retriever(retriever, file_path, chunk_size):
    """ Run the benchmark for a specific retriever, file path, and chunk size. """
    print(f"\n--- Running {retriever} | File: {os.path.basename(file_path)} | Chunk Size: {chunk_size} ---")
    benchmark_retriever(
        dataset_name="niah",
        file_path=file_path,
        retriever_name=retriever,
        embedding_model=EMBEDDING_MODEL,
        num_samples=SAMPLES,
        gpu=GPU,
        chunk_size=chunk_size
    )


def main():
    for file_path in FILE_PATHS:
        for retriever in RETRIEVERS:
            for chunk_size in CHUNK_SIZES:
                run_retriever(retriever, file_path, chunk_size)


if __name__ == "__main__":
    main()
