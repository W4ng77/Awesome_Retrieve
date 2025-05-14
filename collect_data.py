import os
import json
import pandas as pd

INPUT_DIR = "/home/xinyu/Awesome_Retrieve/benchmark_outputs_niah_topk_2_corrected/"
OUTPUT_FILE = "/home/xinyu/Awesome_Retrieve/average_stats.xlsx"

def extract_info_from_filename(filename):
    """
    Extracts retriever, chunk_size, data_type, and top_k from the filename.
    Example:
    - lsh_nmslib_32_synthetic_data_single_qkv_numbers_4.json
    - bm25_64_synthetic_data_single_qkv_words_2.json
    """
    parts = filename.split('_')

    # Extract top_k as the last numeric part
    top_k = parts[-1].split('.')[0]

    # Identify retriever
    if parts[0] == "lsh":
        retriever = f"{parts[0]}_{parts[1]}"
        chunk_size = parts[2]
    else:
        retriever = parts[0]
        chunk_size = parts[1]

    data_type = parts[-2]

    return retriever, chunk_size, data_type, top_k

def process_file(file_path):
    """
    Extracts the "average" entry from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract "average" entry
    avg_entry = next((item for item in data if item["sample_idx"] == "average"), None)
    if not avg_entry:
        return None

    return avg_entry

def collect_data(directory):
    records = []

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(directory, filename)

        # Extract file info
        retriever, chunk_size, data_type, top_k = extract_info_from_filename(filename)

        # Extract average data
        avg_entry = process_file(file_path)
        if avg_entry:
            record = {
                "retriever": retriever,
                "chunk_size": int(chunk_size),
                "data_type": data_type,
                "top_k": int(top_k),
                "accuracy": avg_entry["accuracy"],
                "embedding_time": avg_entry["embedding_time"],
                "indexing_time": avg_entry["indexing_time"],
                "query_embed_time": avg_entry["query_embed_time"],
                "search_time": avg_entry["search_time"]
            }
            records.append(record)

    return records

def save_to_excel(records, output_path):
    """
    Saves the collected records to an Excel file, sorted by specified columns.
    """
    df = pd.DataFrame(records)

    # Sort by retriever, chunk_size, data_type, top_k
    df = df.sort_values(by=["retriever", "chunk_size", "data_type", "top_k"])

    # Save to Excel
    df.to_excel(output_path, index=False)
    print(f"Data saved to {output_path}")

def main():
    # Collect data
    records = collect_data(INPUT_DIR)

    # Save to Excel
    save_to_excel(records, OUTPUT_FILE)

if __name__ == "__main__":
    main()
