import os
import glob
import json
import pandas as pd

# 输入文件夹
input_dir = "benchmark_outputs"

# 加载所有json文件
all_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

# 结果
results_by_chunk = {16: [], 32: [], 64: [], 128: []}
chunking_times = []

for file in all_files:
    filename = os.path.basename(file)
    retriever = filename.split("_hotpotqa")[0]
    chunk_size = int(filename.split("chunk")[1].split("_")[0])

    with open(file, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        # 过滤
        samples = [d for d in data if "sample_idx" in d]
        chunking_infos = [d for d in data if "chunking_total_time" in d]

        if not samples:
            print(f"⚠️ Warning: No samples in {filename}")
            continue

        # 统计四种时间均值
        avg_embed_time = sum(d["embedding_time"] for d in samples) / len(samples)
        avg_index_time = sum(d["indexing_time"] for d in samples) / len(samples)
        avg_query_embed_time = sum(d["query_embed_time"] for d in samples) / len(samples)
        avg_search_time = sum(d["search_time"] for d in samples) / len(samples)

        row = {
            "Retriever": retriever,
            "Embedding Time (s)": round(avg_embed_time, 4),
            "Indexing Time (s)": round(avg_index_time, 4),
            "Query Embed Time (s)": round(avg_query_embed_time, 4),
            "Search Time (s)": round(avg_search_time, 4)
        }

        results_by_chunk[chunk_size].append(row)

        # 只记录一次 chunking 时间（每个retriever一个）
        if chunking_infos:
            chunking_times.append({
                "Retriever": retriever,
                "Chunk Size": chunk_size,
                "Chunking Total Time (s)": round(chunking_infos[0]["chunking_total_time"], 4),
                "Avg Time per Example (ms)": round(chunking_infos[0]["chunking_avg_time_per_example"], 2),
                "Total Examples": chunking_infos[0]["total_examples_processed"]
            })

# 保存每个chunk_size单独的benchmark表
output_dir = "benchmark_summaries"
os.makedirs(output_dir, exist_ok=True)

for chunk_size, rows in results_by_chunk.items():
    if not rows:
        print(f"⚠️ Warning: No data for chunk_size={chunk_size}, skipping...")
        continue

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Retriever")
    output_file = os.path.join(output_dir, f"benchmark_times_chunk{chunk_size}.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")

# 保存chunking时间的总表
if chunking_times:
    chunking_df = pd.DataFrame(chunking_times)
    chunking_df = chunking_df.sort_values(by=["Chunk Size", "Retriever"])
    chunking_output = os.path.join(output_dir, "benchmark_chunking_times.csv")
    chunking_df.to_csv(chunking_output, index=False)
    print(f"✅ Saved chunking table: {chunking_output}")
