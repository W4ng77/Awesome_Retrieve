### 📄 `README.md`

```markdown
# 🔍 Retrieval Benchmarking on HotpotQA

This project benchmarks multiple dense/sparse retrievers using contextual chunking and semantic embeddings over the HotpotQA dataset.

## 📦 Environment Setup

To set up the environment using Conda:

```bash
conda env create -f environment.yaml
conda activate retrieve
````


## 🚀 Benchmarking Usage

### 🔹 Benchmark a Single Retriever

Use `run_benchmark.py` to test one retriever:

```bash
python run_benchmark.py \
  --retriever lsh_faiss \
  --dataset hotpotqa \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --samples 200 \
  --chunk_size 16 \
  --gpu
```

Arguments:

* `--retriever`: choose from `[bm25, dot, faiss, lsh_faiss, lsh_annoy, lsh_nmslib]`
* `--embedding_model`: HuggingFace model name (e.g., MiniLM)
* `--samples`: number of QA samples to test
* `--chunk_size`: number of tokens per passage chunk
* `--gpu`: enable GPU (default is off)

### 🔹 Run All Retrievers

To run all retrievers defined in `run_all.py`:

```bash
python run_all.py
```

This will iterate through all retrievers and chunk sizes, and save results for each.

## 📊 Output Format

All benchmark results are saved in `benchmark_outputs/` as JSON files.

Each sample result looks like this:

```json
{
  "sample_idx": 0,
  "queries": 10,
  "context_chunks": 7200,
  "f1_top10": 0.1084,
  "embedding_time": 1.23,
  "indexing_time": 0.05,
  "query_embed_time": 0.11,
  "search_time": 0.0023
}
```

## 🧩 Project Structure

```
.
├── run_benchmark.py         # Benchmark single retriever
├── run_all.py               # Batch benchmark all retrievers
├── retrievers/              # Contains FAISS, LSH, BM25 retrievers
├── dataset/                 # HotpotQA loading and chunking
├── benchmark_outputs/       # JSON logs for benchmarking results
└── environment.yaml         # Conda environment file
```

## 📄 License

This project is licensed under the MIT License.

---

Pull requests and feedback are welcome!