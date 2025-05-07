from datasets import load_dataset
import time
from transformers import AutoTokenizer


def chunk_text(text, tokenizer, max_tokens):
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


def load_babilong_qa1(
    length_split="128k",
    num_samples=2000,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=16
):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    ds = load_dataset("RMT-team/babilong", length_split, split="qa1")
    ds = ds.select(range(min(num_samples, len(ds))))

    concatenated_data = []
    total_chunk_time = 0.0
    total_chunk_calls = 0

    for example in ds:
        query = example["question"]
        context = example["input"]
        targets = example["target"]  # Updated to "target"

        # ===== 记录切分时间 =====
        t0 = time.time()
        context_chunks = chunk_text(context, tokenizer, max_tokens=chunk_size)
        t1 = time.time()
        total_chunk_time += (t1 - t0)
        total_chunk_calls += 1

        concatenated_entry = {
            "queries": [query],
            "answers": [targets],
            "context": context_chunks
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
        "chunking_avg_time_per_example": round(avg_time_per_example * 1000, 2),
        "total_examples_processed": total_chunk_calls
    }

    return concatenated_data, chunking_info


if __name__ == "__main__":
    length_splits = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M"]
    for length_split in length_splits:
        print(f"\nTesting length split: {length_split}")
        data, chunking_info = load_babilong_qa1(length_split=length_split, num_samples=100)
        print("Loaded data sample:", data[0])
        print("Chunking info:", chunking_info)