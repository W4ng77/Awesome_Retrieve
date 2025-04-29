from datasets import load_dataset
from transformers import AutoTokenizer

def chunk_text(text, tokenizer, max_tokens=16):
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=False)
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        if i >= len(offsets):
            break
        start = offsets[i][0]
        end = offsets[min(i + max_tokens - 1, len(offsets) - 1)][1]
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

import time

def load_hotpotqa_concatenated(
    split="test",
    num_samples=2000,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=16,
    target_token_length=128 * 1024,
):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    ds = load_dataset("THUDM/LongBench", "hotpotqa", split=split)
    ds = ds.select(range(min(num_samples, len(ds))))

    concatenated_data = []
    current_context_chunks = []
    current_total_tokens = 0
    current_queries = []
    current_answers = []

    total_chunk_time = 0.0
    total_chunk_calls = 0

    for example in ds:
        query = example["input"]
        context = example["context"]
        answers = example["answers"]

        # ===== 记录切分时间 =====
        t0 = time.time()
        context_chunks = chunk_text(context, tokenizer, max_tokens=chunk_size)
        t1 = time.time()
        total_chunk_time += (t1 - t0)
        total_chunk_calls += 1

        context_token_count = len(context_chunks) * chunk_size

        if current_total_tokens + context_token_count > target_token_length and current_context_chunks:
            concatenated_entry = {
                "queries": current_queries,
                "answers": current_answers,
                "context": current_context_chunks
            }
            concatenated_data.append(concatenated_entry)

            current_context_chunks = []
            current_total_tokens = 0
            current_queries = []
            current_answers = []

        current_context_chunks.extend(context_chunks)
        current_total_tokens += context_token_count
        current_queries.append(query)
        current_answers.append(answers)

    if current_context_chunks:
        concatenated_entry = {
            "queries": current_queries,
            "answers": current_answers,
            "context": current_context_chunks
        }
        concatenated_data.append(concatenated_entry)

    # ===== 打印切分时间统计 =====
    avg_time_per_example = total_chunk_time / total_chunk_calls if total_chunk_calls else 0
    print(f"\n[Chunking Time Stats]")
    print(f"  Total chunk_text time: {total_chunk_time:.4f} seconds")
    print(f"  Total examples processed: {total_chunk_calls}")
    print(f"  Avg time per example: {avg_time_per_example*1000:.2f} ms\n")
    chunking_info = {
    "chunking_total_time": round(total_chunk_time, 4),
    "chunking_avg_time_per_example": round(avg_time_per_example * 1000, 2),  # ms
    "total_examples_processed": total_chunk_calls
    }
    return concatenated_data, chunking_info
