import json
from run_benchmark import benchmark_retriever

if __name__ == "__main__":
    with open("register.json") as f:
        configs = json.load(f)

    for config in configs:
        benchmark_retriever(
            dataset_name=config["dataset"],
            retriever_name=config["retriever"],
            embedding_model=config.get("embedding_model", ""),
            use_gpu=config.get("gpu", False),
            num_samples=200
        )
