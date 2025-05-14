# Benchmarking Retrievers on HotpotQA

This repository provides benchmarking scripts for evaluating the performance of multiple retrieval methods on the HotpotQA dataset. The benchmark includes F1 score calculation, embedding time, indexing time, query embedding time, and search time for each retriever across different chunk sizes.

## 📦 Files

- `run_niah.py`: Executes benchmarks for the NIAH dataset using synthetic data, supporting multiple retrievers, chunk sizes, and top_k options.
- `run_all_niah.py`: Batch processes all benchmarks for the NIAH dataset with specified configurations.


- `run_benchmark.py`: Executes benchmarking for a specified retriever with defined parameters such as chunk size, embedding model, and sample size.
- `run_all.py`: Executes the benchmarking process for all defined retrievers and chunk sizes, allowing batch processing of all benchmarks.

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Dataset Preparation

Ensure that the HotpotQA dataset is available in the appropriate directory structure and accessible by the script. The dataset can be prepared using the `load_hotpotqa_subset` or `load_hotpotqa_concatenated` functions.

## 🚀 Usage

### 1. NIAH Dataset Preparation

The NIAH dataset is available on Hugging Face and can be downloaded using the following commands:

```python
from datasets import load_dataset

# Download the dataset
dataset = load_dataset("W4ng1204/synthetic_data")

# Save the dataset locally
dataset.save_to_disk("./synthetic_data")
```

To load the dataset from the local directory and save each split as a separate JSONL file:

```python
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("./synthetic_data")

# Save each split as a separate JSONL file
for split in dataset.keys():
    dataset[split].to_json(f"synthetic_data_{split}.jsonl")
```

### 2. Running Benchmarks for NIAH Dataset

To run benchmarks for the NIAH dataset using various retrievers, chunk sizes, and top_k options, execute the following command:

```bash
python run_niah.py
```

This will generate benchmark results for each combination of parameters and save them in the `benchmark_outputs_niah_topk_2_corrected` directory.

Example command for running a single configuration:

```bash
python run_niah.py --retriever faiss --file_path ./synthetic_data/synthetic_data_train.jsonl --chunk_size 32 --top_k 8
```


### 1. Running a Single Benchmark
To run a single benchmark for a specific retriever and chunk size, use the following command:

```bash
python run_benchmark.py --retriever <RETRIEVER_NAME> --embedding_model <MODEL_NAME> --samples <SAMPLE_COUNT> --chunk_size <CHUNK_SIZE> --gpu
```

Example:

```bash
python run_benchmark.py --retriever faiss --embedding_model sentence-transformers/all-MiniLM-L6-v2 --samples 200 --chunk_size 16 --gpu
```

### 2. Running All Benchmarks
To run benchmarks for all defined retrievers and chunk sizes, execute:

```bash
python run_all.py
```

### 3. Output Structure
- Results for each benchmark run are saved in the `benchmark_outputs` directory as JSON files.
- The JSON files contain detailed timing metrics, F1 scores, and chunking information.

Example output structure:
```
benchmark_outputs/
    faiss_hotpotqa_chunk16_results.json
    bm25_hotpotqa_chunk16_results.json
```

## 🛠️ Configuration
Modify the following parameters in `run_all.py` to adjust the benchmarking scope:

- `retrievers`: List of retrievers to benchmark
- `chunk_sizes`: List of chunk sizes to test
- `embedding_model`: Embedding model to use for retrieval
- `samples`: Number of samples to benchmark
- `use_gpu`: Enable/disable GPU usage

## ✅ Results Analysis
- The output JSON files contain metrics such as `f1_top10`, `embedding_time`, `indexing_time`, `query_embed_time`, and `search_time`.
- You can aggregate and analyze the results using tools like pandas or by modifying the `run_all.py` script to generate a summary CSV file.

Example analysis command:

```bash
python -c 'import pandas as pd; df = pd.read_json("benchmark_outputs/faiss_hotpotqa_chunk16_results.json"); print(df.head())'
```

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for more information.
