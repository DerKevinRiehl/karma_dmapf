import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

folder = "../logs_kevin/"

grid_sizes = ["5", "10", "15", "20"]

controllers = [
    "DECENTRALIZED_RESPECT",
    "DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    "DECENTRALIZED_NEGOTIATE_EGOISTIC",
]

controller_labels = [
    "Token Passing",
    "Altruistic",
    "Egoistic",
]

controller_colors = [
    "dodgerblue",
    "green",
    "olive",
]

row_measures = [
    "Completed Tasks",
    "A* Calls",
    "Avg Task Time (incl. Reallocation) (all agents)",
    "Avg Service Time (all agents)",
]

row_labels = [
    "Completion\n[# Tasks]",
    "Runtime\n[# A* Calls]",
    "Avg. Task\nTime [s]",
    "Avg. Service\nTime [s]",
]

# measures = [
#     'A* Calls',
#     'Completed Tasks',
#     'Total Task Time (incl. Reallocation) (all agents)',
#     'Avg Task Time (incl. Reallocation) (all agents)',
#     'Std Task Time (incl. Reallocation) (all agents)',
#     'Total Service Time (all agents)',
#     'Avg Service Time (all agents)',
#     'Std Service Time (all agents)',
#     'Avg Service Time Increase (%) (all agents)',
#     'Avg Service Time (per agent mean)',
#     'Avg Service Increase (%) (per agent mean)'
# ]


def load_data(grid_size, controller, measure):
    # grid_size = "5"
    # controller = "DECENTRALIZED_RESPECT"
    # measure = "Completed Tasks"
    with open(folder + f"summary_{grid_size}_{controller}.json", "r") as file:
        data_dict = json.load(file)
    num_agents = list(data_dict[grid_size][controller]["raw_data"].keys())
    df = []
    for n_agent in num_agents:
        mean = np.mean(data_dict[grid_size][controller]["raw_data"][n_agent][measure])
        std = np.std(data_dict[grid_size][controller]["raw_data"][n_agent][measure])
        df.append([n_agent, mean, std])
    df = pd.DataFrame(df, columns=["n", "mean", "std"])
    df["mean"] = df["mean"].rolling(window=3, center=True, min_periods=1).median()
    df["std"] = df["std"].rolling(window=3, center=True, min_periods=1).median()
    return df


df_respect = load_data(
    grid_size="5", controller="DECENTRALIZED_RESPECT", measure=row_measures[0]
)
df_neg_alt = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[0],
)
df_neg_ego = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[0],
)
controller_dfs_5_rw1 = [df_respect, df_neg_alt, df_neg_ego]
df_respect = load_data(
    grid_size="10", controller="DECENTRALIZED_RESPECT", measure=row_measures[0]
)
df_neg_alt = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[0],
)
df_neg_ego = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[0],
)
controller_dfs_10_rw1 = [df_respect, df_neg_alt, df_neg_ego]

df_respect = load_data(
    grid_size="5", controller="DECENTRALIZED_RESPECT", measure=row_measures[1]
)
df_neg_alt = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[1],
)
df_neg_ego = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[1],
)
controller_dfs_5_rw2 = [df_respect, df_neg_alt, df_neg_ego]
df_respect = load_data(
    grid_size="10", controller="DECENTRALIZED_RESPECT", measure=row_measures[1]
)
df_neg_alt = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[1],
)
df_neg_ego = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[1],
)
controller_dfs_10_rw2 = [df_respect, df_neg_alt, df_neg_ego]

df_respect = load_data(
    grid_size="5", controller="DECENTRALIZED_RESPECT", measure=row_measures[2]
)
df_neg_alt = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[2],
)
df_neg_ego = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[2],
)
controller_dfs_5_rw3 = [df_respect, df_neg_alt, df_neg_ego]
df_respect = load_data(
    grid_size="10", controller="DECENTRALIZED_RESPECT", measure=row_measures[2]
)
df_neg_alt = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[2],
)
df_neg_ego = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[2],
)
controller_dfs_10_rw3 = [df_respect, df_neg_alt, df_neg_ego]

df_respect = load_data(
    grid_size="5", controller="DECENTRALIZED_RESPECT", measure=row_measures[3]
)
df_neg_alt = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[3],
)
df_neg_ego = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[3],
)
controller_dfs_5_rw4 = [df_respect, df_neg_alt, df_neg_ego]
df_respect = load_data(
    grid_size="10", controller="DECENTRALIZED_RESPECT", measure=row_measures[3]
)
df_neg_alt = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[3],
)
df_neg_ego = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[3],
)
controller_dfs_10_rw4 = [df_respect, df_neg_alt, df_neg_ego]


plt.figure(figsize=(16.0, 6.0))

plt.subplot(4, 4, 1)
plt.title("5x5 Grid", fontweight="bold")
plt.ylabel(row_labels[0])
plt.xlim(1, 10 - 1)
plt.xticks([])
for idx, controller_df in enumerate(controller_dfs_5_rw1):
    plt.plot(
        controller_df["n"],
        controller_df["mean"],
        label=controller_labels[idx],
        color=controller_colors[idx],
    )
    plt.fill_between(
        controller_df["n"],
        controller_df["mean"] - controller_df["std"],
        controller_df["mean"] + controller_df["std"],
        color=controller_colors[idx],
        alpha=0.1,
    )
plt.legend(fontsize="small")
plt.margins(x=0)

plt.subplot(4, 4, 2)
plt.title("10x10 Grid", fontweight="bold")
plt.tick_params(axis="x", which="both", labelbottom=False)
plt.xticks([])
for idx, controller_df in enumerate(controller_dfs_10_rw1):
    plt.plot(
        controller_df["n"],
        controller_df["mean"],
        label=controller_labels[idx],
        color=controller_colors[idx],
    )
    plt.fill_between(
        controller_df["n"],
        controller_df["mean"] - controller_df["std"],
        controller_df["mean"] + controller_df["std"],
        color=controller_colors[idx],
        alpha=0.1,
    )

plt.subplot(4, 4, 3)
plt.title("15x15 Grid", fontweight="bold")
plt.xlim(1, 80 - 1)
plt.tick_params(axis="x", which="both", labelbottom=False)
plt.xticks([])

plt.subplot(4, 4, 4)
plt.title("20x20 Grid", fontweight="bold")
plt.xlim(1, 180 - 1)
plt.tick_params(axis="x", which="both", labelbottom=False)
plt.xticks([])


plt.subplot(4, 4, 1 + 4)
plt.ylabel(row_labels[1])
plt.xlim(1, 10 - 1)
plt.xticks([])
for idx, controller_df in enumerate(controller_dfs_5_rw2):
    plt.plot(
        controller_df["n"],
        controller_df["mean"],
        label=controller_labels[idx],
        color=controller_colors[idx],
    )
    plt.fill_between(
        controller_df["n"],
        controller_df["mean"] - controller_df["std"],
        controller_df["mean"] + controller_df["std"],
        color=controller_colors[idx],
        alpha=0.1,
    )

plt.subplot(4, 4, 2 + 4)
plt.tick_params(axis="x", which="both", labelbottom=False)
plt.xticks([])
for idx, controller_df in enumerate(controller_dfs_10_rw2):
    plt.plot(
        controller_df["n"],
        controller_df["mean"],
        label=controller_labels[idx],
        color=controller_colors[idx],
    )
    plt.fill_between(
        controller_df["n"],
        controller_df["mean"] - controller_df["std"],
        controller_df["mean"] + controller_df["std"],
        color=controller_colors[idx],
        alpha=0.1,
    )

plt.subplot(4, 4, 3 + 4)
plt.xlim(1, 80 - 1)
plt.tick_params(axis="x", which="both", labelbottom=False)
plt.xticks([])

plt.subplot(4, 4, 4 + 4)
plt.xlim(1, 180 - 1)
plt.tick_params(axis="x", which="both", labelbottom=False)
plt.xticks([])


plt.subplot(4, 4, 1 + 8)
plt.ylabel(row_labels[2])
plt.xlim(1, 10 - 1)
plt.xticks([])
for idx, controller_df in enumerate(controller_dfs_5_rw3):
    plt.plot(
        controller_df["n"],
        controller_df["mean"],
        label=controller_labels[idx],
        color=controller_colors[idx],
    )
    plt.fill_between(
        controller_df["n"],
        controller_df["mean"] - controller_df["std"],
        controller_df["mean"] + controller_df["std"],
        color=controller_colors[idx],
        alpha=0.1,
    )
plt.margins(x=0)

plt.subplot(4, 4, 2 + 8)
plt.tick_params(axis="x", which="both", labelbottom=False)
plt.xticks([])
for idx, controller_df in enumerate(controller_dfs_10_rw3):
    plt.plot(
        controller_df["n"],
        controller_df["mean"],
        label=controller_labels[idx],
        color=controller_colors[idx],
    )
    plt.fill_between(
        controller_df["n"],
        controller_df["mean"] - controller_df["std"],
        controller_df["mean"] + controller_df["std"],
        color=controller_colors[idx],
        alpha=0.1,
    )

plt.subplot(4, 4, 3 + 8)
plt.xlim(1, 80 - 1)
plt.tick_params(axis="x", which="both", labelbottom=False)
plt.xticks([])
plt.subplot(4, 4, 4 + 8)
plt.xlim(1, 180 - 1)
plt.tick_params(axis="x", which="both", labelbottom=False)
plt.xticks([])


plt.subplot(4, 4, 1 + 12)
plt.xlabel("Density [#Agents]")
plt.ylabel(row_labels[3])
plt.xlim(1, 10 - 1)
for idx, controller_df in enumerate(controller_dfs_5_rw4):
    plt.plot(
        controller_df["n"],
        controller_df["mean"],
        label=controller_labels[idx],
        color=controller_colors[idx],
    )
    plt.fill_between(
        controller_df["n"],
        controller_df["mean"] - controller_df["std"],
        controller_df["mean"] + controller_df["std"],
        color=controller_colors[idx],
        alpha=0.1,
    )

plt.subplot(4, 4, 2 + 12)
plt.xlabel("Density [#Agents]")
# plt.xlim(1,30-1)
# plt.tick_params(axis='x', which='both', labelbottom=False)
for idx, controller_df in enumerate(controller_dfs_10_rw4):
    plt.plot(
        controller_df["n"],
        controller_df["mean"],
        label=controller_labels[idx],
        color=controller_colors[idx],
    )
    plt.fill_between(
        controller_df["n"],
        controller_df["mean"] - controller_df["std"],
        controller_df["mean"] + controller_df["std"],
        color=controller_colors[idx],
        alpha=0.1,
    )

plt.subplot(4, 4, 3 + 12)
plt.xlabel("Density [#Agents]")
plt.xlim(1, 80 - 1)

plt.subplot(4, 4, 4 + 12)
plt.xlabel("Density [#Agents]")
plt.xlim(1, 180 - 1)

plt.tight_layout()
plt.show()

# plt.savefig("Figure_1.pdf")
# plt.savefig("Figure_1.png", dpi=500)
