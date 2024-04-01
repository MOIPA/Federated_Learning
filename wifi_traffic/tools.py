import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import numpy as np

torch.manual_seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def exponential_decay(x, a=0.8):
    return np.exp(-a * x)


def plot_accuracy_vs_epoch(
    accuracies, individual_accuracies, num_epochs, learning_rate
):
    # 定义一个函数，将y轴刻度转换为百分比格式，保留两位小数
    def to_percent(y, position):
        return f"{100*y:.2f}%"

    formatter = FuncFormatter(to_percent)

    # 预定义颜色和标记列表
    colors = ["red", "green", "blue", "purple", "orange", "cyan", "brown", "pink"]
    markers = ["o", "s", "^", "v", "p", "*", "+", "x"]

    plt.figure(figsize=(12, 8))  # Set the figure size to be larger
    plt.plot(
        range(1, num_epochs + 1),
        accuracies,
        marker="o",
        linestyle="-",
        color="black",
        linewidth=3,  # Make the line wider
        label="Server model based on reputation aggregation ",
    )  # Plot accuracy of the aggregated model

    # Plot accuracy of each individual model
    for i, acc_list in enumerate(
        zip(*individual_accuracies)
    ):  # Unpack the nested lists
        label = f"Local GNN model {i+1} "
        plt.plot(
            range(1, num_epochs + 1),
            acc_list,
            marker=markers[i],  # Choose the marker based on node index
            linestyle="--",  # Use dashed line
            color=colors[i],  # Choose the color based on node index
            linewidth=1,  # Make the line thinner
            label=label,
            alpha=0.5,
        )

    plt.xlabel("Epoch", fontsize=14)  # Set the label for the x-axis
    plt.ylabel("Accuracy", fontsize=14)  # Set the label for the y-axis
    plt.title("Model Aggregation", fontsize=16)  # Set the title
    plt.grid(True)  # Add grid lines

    plt.xticks(fontsize=12)  # Set the size of the x-axis ticks
    plt.yticks(fontsize=12)  # Set the size of the y-axis ticks
    plt.gca().yaxis.set_major_formatter(formatter)  # Set the formatter for the y-axis
    plt.xlim([1, num_epochs])  # Set the range of the x-axis
    plt.ylim([0.4, 1])  # Set the range of the y-axis
    plt.legend(loc="lower left", fontsize=12)
    # Find the maximum accuracy and its corresponding epoch
    max_accuracy = max(accuracies)
    max_epoch = accuracies.index(max_accuracy) + 1

    # Print the coordinates of the maximum point
    print(
        f"learning rate {learning_rate}, epoch {num_epochs} ,Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}"
    )

    # # Annotate the maximum point
    # plt.annotate(
    #     f"Max Accuracy: {100*max_accuracy:.2f}%",
    #     xy=(max_epoch, max_accuracy),
    #     xytext=(max_epoch, max_accuracy - 0.1),
    #     arrowprops=dict(facecolor="red", shrink=0.1),
    # )
    plt.tight_layout()
    # plt.savefig("plot.eps", format="eps")
    plt.savefig("plot.png", format="png")
