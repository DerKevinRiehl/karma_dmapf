import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

folder = "../log_files/analysis_1/"
FIGURE_WIDTH = 6.0*2
FIGURE_HEIGHT = 8.0 

controller_labels = [
    "Token Passing",
    "Egoistic",
    "Altruistic",
    "Karma"
]

controller_colors = [
    "dodgerblue",
    "olive",
    "green",
    "red",
]

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

factors = [
    10/4,
    10,
    10
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


def load_data(grid_size, controller, measure, factor=1):
    # grid_size = "5"
    # controller = "DECENTRALIZED_RESPECT"
    # measure = "Completed Tasks"
    with open(folder + f"summary_{controller}_{grid_size}.json", "r") as file:
        data_dict = json.load(file)
    num_agents = list(data_dict[grid_size][controller]["raw_data"].keys())
    df = []
    for n_agent in num_agents:
        mean = np.mean(data_dict[grid_size][controller]["raw_data"][n_agent][measure])
        std = np.std(data_dict[grid_size][controller]["raw_data"][n_agent][measure])
        df.append([n_agent, mean, std])
    df = pd.DataFrame(df, columns=["n", "mean", "std"])
    df["mean"] = df["mean"].rolling(window=3, center=True, min_periods=1).median()*factor
    df["std"] = df["std"].rolling(window=3, center=True, min_periods=1).median()*factor
    return df


df_respect = load_data(
    grid_size="5", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[0], 
    factor=factors[0]
)
df_neg_ego = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[0], 
    factor=factors[0]
)
df_neg_alt = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[0], 
    factor=factors[0]
)
df_neg_kar = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[0], 
    factor=factors[0]
)
controller_dfs_5_rw1 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]
df_respect = load_data(
    grid_size="10", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[0],  
    factor=factors[1]
)
df_neg_ego = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[0],  
    factor=factors[1]
)
df_neg_alt = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[0],  
    factor=factors[1]
)
df_neg_kar = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[0],  
    factor=factors[1]
)
controller_dfs_10_rw1 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]
df_respect = load_data(
    grid_size="15", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[0],  
    factor=factors[1]
)
df_neg_ego = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[0],  
    factor=factors[1]
)
df_neg_alt = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[0],  
    factor=factors[1]
)
df_neg_kar = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[0],  
    factor=factors[1]
)
controller_dfs_15_rw1 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]

df_respect = load_data(
    grid_size="5", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[1],   
    factor=factors[0]
)
df_neg_ego = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[1],   
    factor=factors[0]
)
df_neg_alt = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[1],   
    factor=factors[0]
)
df_neg_kar = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[1],   
    factor=factors[0]
)
controller_dfs_5_rw2 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]
df_respect = load_data(
    grid_size="10", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[1],    
    factor=factors[1]
)
df_neg_ego = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[1],    
    factor=factors[1]
)
df_neg_alt = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[1],    
    factor=factors[1]
)
df_neg_kar = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[1],    
    factor=factors[1]
)
controller_dfs_10_rw2 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]
df_respect = load_data(
    grid_size="15", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[1],   
    factor=factors[0]
)
df_neg_ego = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[1],   
    factor=factors[0]
)
df_neg_alt = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[1],   
    factor=factors[0]
)
df_neg_kar = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[1],   
    factor=factors[0]
)
controller_dfs_15_rw2 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]

df_respect = load_data(
    grid_size="5", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[2],     
)
df_neg_ego = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[2],     
)
df_neg_alt = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[2],     
)
df_neg_kar = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[2],     
)
controller_dfs_5_rw3 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]
df_respect = load_data(
    grid_size="10", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[2],      
)
df_neg_ego = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[2],     
)
df_neg_alt = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[2],      
)
df_neg_kar = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[2],      
)
controller_dfs_10_rw3 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]
df_respect = load_data(
    grid_size="15", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[2],     
)
df_neg_ego = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[2],     
)
df_neg_alt = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[2],     
)
df_neg_kar = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[2],     
)
controller_dfs_15_rw3 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]

df_respect = load_data(
    grid_size="5", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[3],     
)
df_neg_ego = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[3],      
)
df_neg_alt = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[3],      
)
df_neg_kar = load_data(
    grid_size="5",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[3],     
)
controller_dfs_5_rw4 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]
df_respect = load_data(
    grid_size="10", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[3],     
)
df_neg_ego = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[3], 
)
df_neg_alt = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[3],
)
df_neg_kar = load_data(
    grid_size="10",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[3], 
)
controller_dfs_10_rw4 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]
df_respect = load_data(
    grid_size="15", 
    controller="DECENTRALIZED_RESPECT", 
    measure=row_measures[3],     
)
df_neg_ego = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_EGOISTIC",
    measure=row_measures[3],      
)
df_neg_alt = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    measure=row_measures[3],      
)
df_neg_kar = load_data(
    grid_size="15",
    controller="DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
    measure=row_measures[3],     
)
controller_dfs_15_rw4 = [df_respect, df_neg_ego, df_neg_alt, df_neg_kar]

plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

plt.subplot(3, 4, 1 + 4 * 0)
plt.title(row_labels[0], fontweight="bold")
plt.ylabel("5x5 Grid", fontweight="bold")
plt.xlim(1, 10 - 1)
plt.xlabel("Density [#Agents]")
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

plt.subplot(3, 4, 2 + 4 * 0)
plt.title(row_labels[1], fontweight="bold")
plt.xlim(1, 10 - 1)
plt.xlabel("Density [#Agents]")
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

plt.subplot(3, 4, 3 + 4 * 0)
plt.title(row_labels[2], fontweight="bold")
plt.xlim(1, 10 - 1)
plt.xlabel("Density [#Agents]")
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

plt.subplot(3, 4, 4 + 4 * 0)
plt.title(row_labels[3], fontweight="bold")
plt.xlim(1, 10 - 1)
plt.xlabel("Density [#Agents]")
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

plt.subplot(3, 4, 1 + 4 * 1)
plt.ylabel("10x10 Grid", fontweight="bold")
# plt.xlim(1, 30 - 1)
plt.xlabel("Density [#Agents]")
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

plt.subplot(3, 4, 2 + 4 * 1)
# plt.xlim(1, 30 - 1)
plt.xlabel("Density [#Agents]")
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

plt.subplot(3, 4, 3 + 4 * 1)
# plt.xlim(1, 30 - 1)
plt.xlabel("Density [#Agents]")
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

plt.subplot(3, 4, 4 + 4 * 1)
# plt.xlim(1, 30 - 1)
plt.xlabel("Density [#Agents]")
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

plt.subplot(3, 4, 1 + 4 * 2)
plt.ylabel("15x15 Grid", fontweight="bold")
plt.xlabel("Density [#Agents]")
for idx, controller_df in enumerate(controller_dfs_15_rw1):
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
    
plt.subplot(3, 4, 2 + 4 * 2)
plt.xlabel("Density [#Agents]")
for idx, controller_df in enumerate(controller_dfs_15_rw2):
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
    
plt.subplot(3, 4, 3 + 4 * 2)
plt.xlabel("Density [#Agents]")
for idx, controller_df in enumerate(controller_dfs_15_rw3):
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
    
plt.subplot(3, 4, 4 + 4 * 2)
plt.xlabel("Density [#Agents]")
for idx, controller_df in enumerate(controller_dfs_15_rw4):
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
    
plt.subplots_adjust(top=0.960, bottom=0.060, left=0.060, right=0.990, hspace=0.250, wspace=0.230)
plt.savefig("Figure_1.png", dpi=300)
plt.show()

# plt.savefig("Figure_1.pdf")
