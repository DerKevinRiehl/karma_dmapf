from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METRIC_NAME = "Avg Service Time Increase (%) (all agents)"
SUMMARY_FILENAME = "summary_grid10_agents30_T100.json"  # keep in sync with generator

CONTROLLER_LABELS = {
    "DECENTRALIZED_RESPECT": "Token Passing",
    "DECENTRALIZED_NEGOTIATE_EGOISTIC": "Egoistic",
    "DECENTRALIZED_NEGOTIATE_ALTRUISTIC": "Altruistic",
    "DECENTRALIZED_NEGOTIATE_TRIP_KARMA": "Trip Karma",
}

CONTROLLER_COLORS = {
    "DECENTRALIZED_RESPECT": "dodgerblue",
    "DECENTRALIZED_NEGOTIATE_EGOISTIC": "olive",
    "DECENTRALIZED_NEGOTIATE_ALTRUISTIC": "green",
    "DECENTRALIZED_NEGOTIATE_TRIP_KARMA": "darkorange",
}


def load_summary() -> tuple[pd.DataFrame, dict]:
    base_dir = Path(__file__).resolve().parent.parent
    summary_path = base_dir / "logs_js" / SUMMARY_FILENAME
    data = json.loads(summary_path.read_text())
    summary_df = pd.DataFrame(data.get("summary", []))
    metadata = data.get("metadata", {})
    return summary_df, metadata


def plot_influence(summary_df: pd.DataFrame, metadata: dict) -> None:
    if summary_df.empty:
        raise ValueError("summary.json is empty; run the analysis script first.")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

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
        )
        if "std" in subset.columns:
            ax.fill_between(
                subset["karma_influence"],
                subset["mean"] - subset["std"],
                subset["mean"] + subset["std"],
                color=color,
                alpha=0.15,
            )

    grid = metadata.get("grid_size_base", "?")
    agents = metadata.get("n_agents", "?")
    ax.set_xlabel("Karma Influence")
    ax.set_ylabel(METRIC_NAME)
    ax.set_title(
        f"Service Time Increase vs. Karma Influence (grid={grid}, agents={agents})"
    )
    ax.legend(title="Controller", fontsize="small")
    plt.tight_layout()
    plt.show()
    fig.savefig("results/Figure_3.pdf", dpi=500)
    fig.savefig("results/Figure_3.png", dpi=500)


if __name__ == "__main__":
    summary_df, metadata = load_summary()
    plot_influence(summary_df, metadata)
