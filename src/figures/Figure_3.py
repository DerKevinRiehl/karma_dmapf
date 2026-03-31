from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


folder = "log_files/analysis_3/"

METRIC_NAME = "Avg Service Time Increase (%) (all agents)"
SUMMARY_FILES = [
    "summary_grid5_agents10_T100.json",
    "summary_grid10_agents30_T100.json",
    "summary_grid15_agents80_T100.json",
]

CONTROLLER_LABELS = {
    "DECENTRALIZED_RESPECT": "Token Passing",
    "DECENTRALIZED_NEGOTIATE_EGOISTIC": "Egoistic",
    "DECENTRALIZED_NEGOTIATE_ALTRUISTIC": "Altruistic",
    "DECENTRALIZED_NEGOTIATE_TRIP_KARMA": "Karma",
}

# Match Figure_2 palette
CONTROLLER_COLORS = {
    "DECENTRALIZED_RESPECT": "dodgerblue",
    "DECENTRALIZED_NEGOTIATE_EGOISTIC": "olive",
    "DECENTRALIZED_NEGOTIATE_ALTRUISTIC": "green",
    "DECENTRALIZED_NEGOTIATE_TRIP_KARMA": "red",
}


def load_summary(summary_filename: str) -> tuple[pd.DataFrame, dict]:
    base_dir = Path(__file__).resolve().parent.parent
    summary_path = base_dir / folder / summary_filename
    data = json.loads(summary_path.read_text())
    summary_df = pd.DataFrame(data.get("summary", []))
    metadata = data.get("metadata", {})
    return summary_df, metadata


def plot_influences(summary_dfs: list[pd.DataFrame], metadatas: list[dict]) -> None:
    if any(df.empty for df in summary_dfs):
        raise ValueError(
            "One or more summaries are empty; run the analysis script first."
        )

    plt.style.use("default")
    fig, axes = plt.subplots(len(summary_dfs), 1, figsize=(6.0, 9.0), sharex=True)
    if len(summary_dfs) == 1:
        axes = [axes]

    for ax, summary_df, metadata in zip(axes, summary_dfs, metadatas):
        for controller, label in CONTROLLER_LABELS.items():
            subset = summary_df[
                (summary_df["controller"] == controller)
                & (summary_df["metric"] == METRIC_NAME)
            ]
            if subset.empty:
                continue
            subset = subset.sort_values("karma_influence")
            color = CONTROLLER_COLORS.get(controller, None)
            ax.plot(
                subset["karma_influence"],
                subset["mean"],
                label=label,
                color=color,
                marker="o",
                linewidth=1.5,
                markersize=4,
            )
            if "std" in subset.columns:
                ax.fill_between(
                    subset["karma_influence"],
                    subset["mean"] - subset["std"],
                    subset["mean"] + subset["std"],
                    color=color,
                    alpha=0.12,
                )

        grid = metadata.get("grid_size_base", "?")
        agents = metadata.get("n_agents", "?")
        ax.set_ylabel("Service Time Increase [%]")
        ax.set_title(
            f"{grid}x{grid} Grid | Agents {agents}", fontsize=10, fontweight="bold"
        )
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Karma Influence")

    # Single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize="small",
        frameon=True,
        edgecolor="lightgray",
        facecolor="white",
        framealpha=1.0,
    )

    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.savefig("results/Figure_3.png", dpi=300)
    fig.savefig("results/Figure_3.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    summary_dfs: list[pd.DataFrame] = []
    metadatas: list[dict] = []
    for fname in SUMMARY_FILES:
        df, meta = load_summary(fname)
        summary_dfs.append(df)
        metadatas.append(meta)
    plot_influences(summary_dfs, metadatas)
