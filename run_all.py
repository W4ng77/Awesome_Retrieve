# -------- File: run_all.py --------
import subprocess
import os
import glob
import pandas as pd

retrievers = [
    "bm25",
    "dot",
    "faiss",
    "lsh_faiss",
    "lsh_annoy",
    "lsh_nmslib",
    
]

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
dataset = "hotpotqa"
samples = 200
use_gpu = True

for retriever in retrievers:
    print(f"\n>>> Running benchmark for: {retriever}")
    cmd = [
        "python", "run_benchmark.py",
        "--dataset", dataset,
        "--retriever", retriever,
        "--embedding_model", embedding_model,
        "--samples", str(samples)
    ]
    if use_gpu and retriever != "bm25":
        cmd.append("--gpu")

    subprocess.run(cmd)

# -------- 合并所有结果为 summary.csv --------
result_files = sorted(glob.glob("benchmark_results_*.csv"))
dfs = [pd.read_csv(f) for f in result_files]
summary_df = pd.concat(dfs, ignore_index=True)
summary_df.to_csv("benchmark_summary.csv", index=False)

print(f"\n✅ All results saved to benchmark_summary.csv with {len(summary_df)} rows.")
