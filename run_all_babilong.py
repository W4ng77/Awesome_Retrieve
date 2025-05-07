import subprocess
import os
import glob
import pandas as pd
import json

retrievers = [
    "bm25",
    "dot",
    "faiss",
    "lsh_faiss",
    "lsh_annoy",
    "lsh_nmslib",
]

chunk_sizes = [16, 32, 64, 128]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
dataset = "babilong_qa1"
samples = 100
use_gpu = True

# Length splits for qa1 in BABILong
length_splits = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M"]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

output_dir = "benchmark_outputs"
os.makedirs(output_dir, exist_ok=True)

for chunk_size in chunk_sizes:
    for retriever in retrievers:
        for length_split in length_splits:
            print(f"\n>>> Running benchmark for: {retriever} | chunk_size: {chunk_size} | length_split: {length_split}")
            
            # Construct the output filename with length_split
            output_filename = f"{retriever}_{dataset}_chunk{chunk_size}_{length_split}_results.json"
            output_path = os.path.join(output_dir, output_filename)

            # Check if the file already exists to avoid re-running
            if os.path.exists(output_path):
                print(f"Skipping {output_path} as it already exists.")
                continue

            cmd = [
                "python", "run_benchmark_babilong.py",
                "--dataset", dataset,
                "--retriever", retriever,
                "--embedding_model", embedding_model,
                "--samples", str(samples),
                "--chunk_size", str(chunk_size),
                "--length_splits", length_split
            ]
            if use_gpu and retriever != "bm25":
                cmd.append("--gpu")

            subprocess.run(cmd)

# -------- Merge All Results into summary.csv --------
result_files = sorted(glob.glob(os.path.join(output_dir, "*.json")))

all_results = []
for file_path in result_files:
    with open(file_path, "r") as f:
        data = json.load(f)
        all_results.extend(data)

# Convert to DataFrame
summary_df = pd.DataFrame(all_results)
summary_csv_path = os.path.join(output_dir, "benchmark_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"\n✅ All results saved to {summary_csv_path} with {len(summary_df)} rows.")
