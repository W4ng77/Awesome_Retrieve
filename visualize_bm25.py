import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
EXCEL_FILE = "/home/xinyu/Awesome_Retrieve/average_stats.xlsx" # Path to your generated Excel file
OUTPUT_PLOT_DIR = "/home/xinyu/Awesome_Retrieve/bm25_topk4_visualizations/"
RETRIEVER_TO_ANALYZE = "bm25"
TOP_K_TO_ANALYZE = 4

# --- Helper Functions ---
def load_and_filter_data(excel_path, retriever_name, top_k_value):
    """
    Loads data from the Excel file and filters it for the specified retriever and top_k.
    """
    try:
        df = pd.read_excel(excel_path)
        print(f"Successfully loaded data from {excel_path}")
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_path}")
        return None

    # Basic filtering
    filtered_df = df[
        (df["retriever"] == retriever_name) &
        (df["top_k"] == top_k_value)
    ]

    if filtered_df.empty:
        print(f"No data found for retriever '{retriever_name}' and top_k={top_k_value}.")
        return None

    print(f"Filtered data for {retriever_name}, top_k={top_k_value}:")
    print(filtered_df.head())
    return filtered_df

def plot_metrics(df, output_dir):
    """
    Generates and saves plots for accuracy, search_time, and indexing_time
    against chunk_size, grouped by data_type.
    """
    if df is None or df.empty:
        print("No data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    # Ensure chunk_size is treated as a category for plotting if it's not continuous,
    # or sort it if it's numeric for line plots.
    # For line plots, numeric chunk_size is better.
    df = df.sort_values(by=["data_type", "chunk_size"])

    # Plot 1: Accuracy vs. Chunk Size
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="chunk_size", y="accuracy", hue="data_type", marker="o", palette="viridis")
    plt.title(f"Accuracy vs. Chunk Size (Retriever: {RETRIEVER_TO_ANALYZE}, Top K: {TOP_K_TO_ANALYZE})")
    plt.xlabel("Chunk Size")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Data Type")
    plot_path = os.path.join(output_dir, "accuracy_vs_chunk_size.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")

    # Plot 2: Search Time vs. Chunk Size
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="chunk_size", y="search_time", hue="data_type", marker="o", palette="viridis")
    plt.title(f"Search Time vs. Chunk Size (Retriever: {RETRIEVER_TO_ANALYZE}, Top K: {TOP_K_TO_ANALYZE})")
    plt.xlabel("Chunk Size")
    plt.ylabel("Search Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Data Type")
    plot_path = os.path.join(output_dir, "search_time_vs_chunk_size.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")

    # Plot 3: Indexing Time vs. Chunk Size
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="chunk_size", y="indexing_time", hue="data_type", marker="o", palette="viridis")
    plt.title(f"Indexing Time vs. Chunk Size (Retriever: {RETRIEVER_TO_ANALYZE}, Top K: {TOP_K_TO_ANALYZE})")
    plt.xlabel("Chunk Size")
    plt.ylabel("Indexing Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Data Type")
    plot_path = os.path.join(output_dir, "indexing_time_vs_chunk_size.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")

    # Optional: Bar plot for average metrics if that makes sense
    # For instance, average accuracy per data_type across chunk_sizes
    avg_accuracy_by_datatype = df.groupby("data_type")["accuracy"].mean().reset_index()
    if not avg_accuracy_by_datatype.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_accuracy_by_datatype, x="data_type", y="accuracy", palette="pastel")
        plt.title(f"Average Accuracy by Data Type (Retriever: {RETRIEVER_TO_ANALYZE}, Top K: {TOP_K_TO_ANALYZE})")
        plt.xlabel("Data Type")
        plt.ylabel("Average Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "avg_accuracy_by_datatype.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")


def main():
    """
    Main function to load, filter, and visualize BM25 data.
    """
    bm25_topk4_data = load_and_filter_data(EXCEL_FILE, RETRIEVER_TO_ANALYZE, TOP_K_TO_ANALYZE)

    if bm25_topk4_data is not None:
        # You can create a new "table" (DataFrame) from this filtered data
        # and save it if needed, e.g., to a new CSV or Excel.
        output_filtered_excel_path = os.path.join(OUTPUT_PLOT_DIR, f"{RETRIEVER_TO_ANALYZE}_topk{TOP_K_TO_ANALYZE}_filtered_data.xlsx")
        os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True) # ensure dir exists before saving excel
        bm25_topk4_data.to_excel(output_filtered_excel_path, index=False)
        print(f"Saved filtered BM25 top_k=4 data to: {output_filtered_excel_path}")

        # Proceed to plotting
        plot_metrics(bm25_topk4_data, OUTPUT_PLOT_DIR)
        print("Visualization complete.")
    else:
        print("Could not proceed with visualization due to data loading/filtering issues.")

if __name__ == "__main__":
    main()