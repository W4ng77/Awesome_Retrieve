import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_FILE = "/home/xinyu/Awesome_Retrieve/average_stats.xlsx"
OUTPUT_DIR = "/home/xinyu/Awesome_Retrieve/plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_combined_accuracy_topk_8(df):
    """
    Plot combined accuracy for top_k = 8 across three data types in a single horizontal layout.
    """
    data_types = ["numbers", "uuids", "words"]
    retrievers = df["retriever"].unique()

    # Create a horizontal layout with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Accuracy vs. Chunk Size (Top-K: 8)", fontsize=16, weight='bold')

    for i, data_type in enumerate(data_types):
        ax = axes[i]
        for retriever in retrievers:
            subset = df[(df["data_type"] == data_type) & (df["top_k"] == 8) & (df["retriever"] == retriever)]
            if subset.empty:
                continue

            ax.plot(subset["chunk_size"], subset["accuracy"], marker='o', label=retriever)

        ax.set_title(f"Data Type: {data_type}")
        ax.set_xlabel("Chunk Size")
        ax.grid(alpha=0.3)
        ax.set_xticks([32, 64, 128])

        # Only set y-axis label on the first subplot
        if i == 0:
            ax.set_ylabel("Accuracy")

    # Add legend to the first subplot
    axes[0].legend(title="Retriever", loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "accuracy_combined_topk_8.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved combined plot: {output_path}")

def main():
    # Load data
    df = pd.read_excel(INPUT_FILE)

    # Plot combined accuracy for top_k = 8
    plot_combined_accuracy_topk_8(df)

if __name__ == "__main__":
    main()
