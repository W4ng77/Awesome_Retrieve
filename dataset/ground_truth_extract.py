import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import torch

def chunk_text(text, tokenizer, max_tokens=64, stride=16):
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=False, padding=False)
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]

    if isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if isinstance(offsets[0], list):
        offsets = offsets[0]

    chunks = []
    for i in range(0, len(input_ids), stride):
        if i >= len(offsets):
            break
        start_offset = offsets[i][0]
        end_idx = min(i + max_tokens - 1, len(offsets) - 1)
        end_offset = offsets[end_idx][1]
        chunk = text[start_offset:end_offset].strip()
        if chunk:
            chunks.append(chunk)
        if i + max_tokens >= len(input_ids):
            break
    return chunks

def retrieve_top_chunks_dense(query, chunks, embedder, top_k=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(chunks)))
    return [(chunks[i], float(top_scores[j])) for j, i in enumerate(top_indices)]

def retrieve_top_chunks_bm25(query, chunks, top_k=3):
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]

def process_longbench_tasks_combined(
    tasks=["hotpotqa", "2wikimqa", "musique", "multifieldqa_en"],
    split="test",
    chunk_sizes=[64],
    stride=16,
    model_name="all-mpnet-base-v2",
    output_dir="longbench_support_facts_combined"
):
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")
    embedder = SentenceTransformer(model_name)
    os.makedirs(output_dir, exist_ok=True)

    for task in tasks:
        print(f"\n🔍 Processing task: {task}")
        try:
            ds = load_dataset("THUDM/LongBench", task, split=split)
        except Exception as e:
            print(f"❌ Failed to load task {task}: {e}")
            continue

        for chunk_size in chunk_sizes:
            print(f"  🔹 Chunk size: {chunk_size}")
            results = []

            for idx, example in enumerate(ds):
                query = example.get("input", "")
                answer_list = example.get("answers", [])
                context = example.get("context", "")
                answer = answer_list[0] if answer_list else ""
                full_query = query + " " + answer

                chunks = chunk_text(context, tokenizer, max_tokens=chunk_size, stride=stride)

                dense_results = retrieve_top_chunks_dense(full_query, chunks, embedder, top_k=3)
                bm25_results = retrieve_top_chunks_bm25(full_query, chunks, top_k=3)

                dense_chunks = [chunk for chunk, _ in dense_results]
                bm25_chunks = [chunk for chunk, _ in bm25_results]

                results.append({
                    "id": idx,
                    "input": query,
                    "answer": answer,
                    "support_facts": {
                        "dense": dense_chunks,
                        "sparse": bm25_chunks
                    }
                })

            output_path = os.path.join(output_dir, f"{task}_chunk{chunk_size}.json")
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved: {output_path}")

# Run the process
process_longbench_tasks_combined()
