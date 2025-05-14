import pandas as pd

INPUT_FILE = "/home/xinyu/Awesome_Retrieve/average_stats.xlsx"

def generate_table_1(df):
    """
    Generate Table 1: data_type - chunk_size - accuracy (Top-K = 4, 8)
    """
    # 只保留 top_k 为 4 和 8 的数据
    subset = df[df["top_k"].isin([4, 8])]

    # 计算平均 accuracy，按 data_type, chunk_size, top_k 分组
    table_1 = (
        subset.groupby(["data_type", "chunk_size", "top_k"])["accuracy"]
        .mean()
        .unstack()
        .reset_index()
    )

    # 重命名列
    table_1.columns = ["data_type", "chunk_size", "avg_accuracy_topk_4", "avg_accuracy_topk_8"]

    # 保存到 Excel
    output_path = "/home/xinyu/Awesome_Retrieve/table_1_accuracy_chunk_size.xlsx"
    table_1.to_excel(output_path, index=False)
    print(f"Table 1 saved to {output_path}")
def generate_table_2(df):
    """
    Generate Table 2: retriever - chunk_size - indexing_time - search_time - embedding_time - query_embed_time (Top-K = 8 only)
    """
    # 只保留 top_k 为 8 的数据
    subset = df[df["top_k"] == 8]

    # 按 retriever, chunk_size 分组，计算平均时间
    table_2 = (
        subset.groupby(["retriever", "chunk_size"])[["indexing_time", "search_time", "embedding_time", "query_embed_time"]]
        .mean()
        .reset_index()
    )

    # 重命名列
    table_2.columns = [
        "retriever", "chunk_size",
        "avg_indexing_time_topk_8", 
        "avg_search_time_topk_8", 
        "avg_embedding_time_topk_8", 
        "avg_query_embed_time_topk_8"
    ]

    # 保存到 Excel
    output_path = "/home/xinyu/Awesome_Retrieve/table_2_times_chunk_size_topk_8.xlsx"
    table_2.to_excel(output_path, index=False)
    print(f"Table 2 saved to {output_path}")
def generate_table_3(df):
    """
    Generate Table 3: retriever - data_type - top_k - accuracy - search_time
    """
    # 按 retriever, data_type, top_k 分组，计算平均 accuracy 和 search_time
    table_3 = (
        df.groupby(["retriever", "data_type", "top_k"])[["accuracy", "search_time"]]
        .mean()
        .reset_index()
    )

    # 保存到 Excel
    output_path = "/home/xinyu/Awesome_Retrieve/table_3_topk_metrics.xlsx"
    table_3.to_excel(output_path, index=False)
    print(f"Table 3 saved to {output_path}")

def main():
    # Load data
    df = pd.read_excel(INPUT_FILE)

    # Generate Table 1
    generate_table_1(df)
    # Generate Table 2
    generate_table_2(df)
    # Generate Table 3
    generate_table_3(df)
if __name__ == "__main__":
    main()
