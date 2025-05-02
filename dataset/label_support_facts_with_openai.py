import os
import json
import hashlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CACHE_FILE = "openai_support_fact_cache.json"
os.makedirs(os.path.dirname(CACHE_FILE) or ".", exist_ok=True)

# 加载缓存
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

def get_cache_key(question, answer, support_fact):
    key_str = f"{question}||{answer}||{support_fact}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

def evaluate_support_fact(question, answer, support_fact, model="gpt-4o"):
    prompt = f"""\
Question: {question}
Answer: {answer}
Support Fact: {support_fact}

Does the support fact help answer the question or support the given answer?

Reply with one word only: YES or NO."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        reply = response.choices[0].message.content.strip().upper()
        return reply == "YES"
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

def process_fact_entry(entry, model="gpt-4o"):
    question, answer, fact, source = entry
    result = evaluate_support_fact(question, answer, fact, model)
    return (source, fact, result)

def label_file(input_path, output_path, model="gpt-4o", max_workers=5):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries_to_process = []
    for item in data:
        q, a = item["input"], item["answer"]
        for fact in item["support_facts"]["dense"]:
            entries_to_process.append((q, a, fact, "dense"))
        for fact in item["support_facts"]["sparse"]:
            entries_to_process.append((q, a, fact, "sparse"))

    labeled_map = {}  # key: (q,a,fact,source) -> result
    print(f"🔄 Submitting {len(entries_to_process)} tasks to OpenAI...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fact_entry, entry, model): entry for entry in entries_to_process}
        for future in tqdm(as_completed(futures), total=len(futures)):
            source, fact, result = future.result()
            entry = futures[future]
            key = (entry[0], entry[1], fact, source)
            labeled_map[key] = result

    # 构造新数据结构
    labeled_data = []
    for item in data:
        q, a = item["input"], item["answer"]
        dense = []
        sparse = []

        for fact in item["support_facts"]["dense"]:
            result = labeled_map.get((q, a, fact, "dense"), False)
            dense.append({"text": fact, "if_ground_truth": result})

        for fact in item["support_facts"]["sparse"]:
            result = labeled_map.get((q, a, fact, "sparse"), False)
            sparse.append({"text": fact, "if_ground_truth": result})

        item["support_facts"] = {
            "dense": dense,
            "sparse": sparse
        }
        labeled_data.append(item)

    # 保存输出
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Labeled file saved to: {output_path}")

    # 保存缓存
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"💾 Cache updated: {CACHE_FILE}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model (default: gpt-4o)")
    parser.add_argument("--max_workers", type=int, default=5, help="Max concurrent threads")
    args = parser.parse_args()

    output_path = args.input.replace(".json", "_labeled.json")
    label_file(args.input, output_path, model=args.model, max_workers=args.max_workers)
