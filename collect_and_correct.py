import os
import json
import pandas as pd

def extract_info_from_filename(filename):
    """
    Extract method, chunk_size, data_type, and top_k from the filename.
    Supports:
    - lsh_faiss_128_synthetic_data_single_qkv_numbers_4.json
    - bm25_64_synthetic_data_single_qkv_words_2.json
    """
    parts = filename.split('_')

    # Extract `top_k` as the last numeric part before `.json`
    top_k = parts[-1].split('.')[0]

    # Method can be a single or multi-part method (e.g., lsh_faiss, bm25)
    if parts[0] == "lsh" and parts[1] == "faiss":
        method = "lsh_faiss"
        chunk_size = parts[2]
        data_type = parts[-2]
    else:
        method = parts[0]
        chunk_size = parts[1]
        data_type = parts[-2]

    return method, chunk_size, data_type, top_k

def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract "average" entry
    avg_entry = next((item for item in data if item["sample_idx"] == "average"), None)
    if not avg_entry:
        return None

    # Extract necessary fields
    accuracy = avg_entry["accuracy"]
    embedding_time = avg_entry["embedding_time"]
    indexing_time = avg_entry["indexing_time"]
    query_embed_time = avg_entry["query_embed_time"]
    search_time = avg_entry["search_time"]

    # Apply corrections
    corrected_indexing_time = indexing_time - embedding_time
    corrected_search_time = (search_time - query_embed_time) / 2

    return {
        "accuracy": accuracy,
        "embedding_time": embedding_time,
        "indexing_time": round(corrected_indexing_time, 5),
        "query_embed_time": query_embed_time,
        "search_time": round(corrected_search_time, 5)
    }

def collect_data(directory):
    records = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            # Skip `annoy` methods
            if "annoy" in filename:
                continue

            file_path = os.path.join(directory, filename)
            
            # Extract method, chunk_size, data_type, top_k
            method, chunk_size, data_type, top_k = extract_info_from_filename(filename)

            # Process and correct the data
            entry = process_file(file_path)
            if entry:
                entry["method"] = method
                entry["chunk_size"] = chunk_size
                entry["data_type"] = data_type
                entry["top_k"] = top_k
                records.append(entry)

    return records

def save_to_excel(records, output_path):
    df = pd.DataFrame(records)
    df.to_excel(output_path, index=False)
    print(f"Data saved to {output_path}")

def main():
    directory = "/home/xinyu/Awesome_Retrieve/benchmark_outputs_niah_topk_2/"
    output_path = "/home/xinyu/Awesome_Retrieve/summary.xlsx"

    # Collect and correct data
    records = collect_data(directory)

    # Save to Excel
    save_to_excel(records, output_path)

if __name__ == "__main__":
    main()
