import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = "../log_files/analysis_1/"
FIGURE_WIDTH = 6.0 * 2
FIGURE_HEIGHT = 5.0

controllers = [
    ("DECENTRALIZED_RESPECT", "Token Passing", "dodgerblue"),
    ("DECENTRALIZED_NEGOTIATE_EGOISTIC", "Egoistic", "olive"),
    ("DECENTRALIZED_NEGOTIATE_ALTRUISTIC", "Altruistic", "green"),
    ("DECENTRALIZED_NEGOTIATE_TRIP_KARMA", "Karma", "red"),
]

grid_sizes = ["5", "10", "15"]
grid_labels = {
    "5": "5x5 Grid",
    "10": "10x10 Grid",
    "15": "15x15 Grid",
}

row_measures = [
    "Completed Tasks",
    "A* Calls",
    "Avg Task Time (incl. Reallocation) (all agents)",
    "Avg Service Time (all agents)",
]

row_labels = [
    "Completion [# Tasks]",
    "Runtime [# A* Calls]",
    "Avg. Task Time [s]",
    "Avg. Service Time [s]",
]

# Preserves the original scaling choices from the previous script.
scale_factors = {
    "5": [10 / 4, 10 / 4, 1, 1],
    "10": [10, 10, 1, 1],
    "15": [10, 10 / 4, 1, 1],
}


def load_data(grid_size, controller, measure, factor=1):
    with open(folder + f"summary_{controller}_{grid_size}.json", "r") as file:
        data_dict = json.load(file)

    raw_data = data_dict[grid_size][controller]["raw_data"]
    rows = []
    for n_agent, values_by_measure in raw_data.items():
        values = values_by_measure[measure]
        rows.append([n_agent, np.mean(values), np.std(values)])

    df = pd.DataFrame(rows, columns=["n", "mean", "std"])
    df["n"] = pd.to_numeric(df["n"])
    df = df.sort_values("n").reset_index(drop=True)
    df["mean"] = df["mean"].rolling(window=3, center=True, min_periods=1).median() * factor
    df["std"] = df["std"].rolling(window=3, center=True, min_periods=1).median() * factor
    return df


def build_plot_data():
    plot_data = {}
    for grid_size in grid_sizes:
        plot_data[grid_size] = []
        for measure_idx, measure in enumerate(row_measures):
            factor = scale_factors[grid_size][measure_idx]
            controller_dfs = [
                load_data(grid_size, controller_name, measure, factor)
                for controller_name, _, _ in controllers
            ]
            plot_data[grid_size].append(controller_dfs)
    return plot_data


def plot_subplot(ax, controller_dfs, title=None, ylabel=None, show_legend=False):
    ax.grid(True, alpha=0.2)
    if title is not None:
        ax.set_title(title, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold")

    ax.set_xlabel("Density [#Agents]")
    ax.set_xlim(controller_dfs[0]["n"].min(), controller_dfs[0]["n"].max())
    ax.margins(x=0)

    for controller_df, (_, label, color) in zip(controller_dfs, controllers):
        ax.plot(
            controller_df["n"],
            controller_df["mean"],
            label=label,
            color=color,
        )
        ax.fill_between(
            controller_df["n"],
            controller_df["mean"] - controller_df["std"],
            controller_df["mean"] + controller_df["std"],
            color=color,
            alpha=0.1,
        )

    if show_legend:
        ax.legend(fontsize="x-small", loc="upper left",
                  frameon=True, framealpha=0)   # transparent box)


plot_data = build_plot_data()

fig, axes = plt.subplots(3, 4, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

for row_idx, grid_size in enumerate(grid_sizes):
    for col_idx, controller_dfs in enumerate(plot_data[grid_size]):
        ax = axes[row_idx, col_idx]
        plot_subplot(
            ax,
            controller_dfs,
            title=row_labels[col_idx] if row_idx == 0 else None,
            ylabel=grid_labels[grid_size] if col_idx == 0 else None,
            show_legend=(row_idx == 0 and col_idx == 0),
        )

plt.subplots_adjust(top=0.950, bottom=0.090, left=0.060, right=0.990, hspace=0.400, wspace=0.230)
plt.savefig("Figure_1.png", dpi=300)
plt.show()
