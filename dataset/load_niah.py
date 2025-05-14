import json
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def chunk_text(text, tokenizer, chunk_size=16, stride=16):
    """
    Tokenizes text and splits into overlapping chunks.
    - chunk_size: Length of each chunk in tokens.
    - stride: Fixed number of tokens to move for each chunk (16 by default).
    """
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=False)
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]
    chunks = []
    length = len(input_ids)
    
    # Iterate with fixed stride of 16
    for start in range(0, length, stride):
        end = start + chunk_size
        
        # Adjust to prevent out-of-bound access
        end = min(end, length)
        
        # Ensure valid range
        if start >= length:
            break
        
        # Extract the text slice using offset positions
        start_pos = offsets[start][0]
        end_pos = offsets[end - 1][1]
        chunk = text[start_pos:end_pos].strip()
        
        if chunk:
            chunks.append(chunk)
    
    return chunks

def load_niah_dataset(file_path, embedding_model="gpt2", chunk_size=16, num_samples=200):
    """
    Load the dataset and process it with overlapping chunking.
    """
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]

    # Limit to the specified number of samples
    data = data[:num_samples]

    processed_data = []
    chunking_info = []
    
    for idx, example in enumerate(tqdm(data, desc="Processing NIAH dataset")):
        # Extract the query
        query = example["input"].split("\\n")[-1].strip()

        # Extract context before the last \\n
        context = example["input"].rsplit("\\n", 1)[0]

        # Ensure trailing period
        if context and not context.endswith("."):
            context += "."

        # Extract the answer
        answer = example["outputs"][0] if isinstance(example["outputs"], list) and example["outputs"] else ""

        # Apply the modified chunking function
        context_chunks = chunk_text(context, tokenizer, chunk_size=chunk_size, stride=16)

        processed_data.append({
            "query": query,
            "context": context_chunks,
            "answer": answer
        })
        
        chunking_info.append({
            "sample_idx": idx,
            "num_chunks": len(context_chunks),
            "queries": 1
        })
    
    return processed_data, chunking_info
