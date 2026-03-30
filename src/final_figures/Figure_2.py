import matplotlib.pyplot as plt
from matplotlib.patches import Patch

folder = "../logs_kevin/"

controller_file_names = [
    "DECENTRALIZED_RESPECT",
    "DECENTRALIZED_NEGOTIATE_EGOISTIC",
    "DECENTRALIZED_NEGOTIATE_ALTRUISTIC",
    "DECENTRALIZED_NEGOTIATE_TRIP_KARMA",
]

controller_labels = [
    "Token Passing",
    "Egoistic",
    "Altruistic",
    "Karma",
]

controller_colors = [
    "dodgerblue",
    "olive",
    "green",
    "red",
]

def load_time_values(ftype, controller, grid_size, n_agents):
    file_path = f"{folder}{ftype}_{controller}_{grid_size}_{n_agents}.txt"
    with open(file_path, "r") as file:
        values = [float(value) for value in file.read().splitlines() if value.strip()]
    return values


def load_grid_data(grid_size, n_agents):
    return {
        "service_times": [
            load_time_values("all_service_times", controller, grid_size, n_agents)
            for controller in controller_file_names
        ],
        "task_times": [
            load_time_values("all_task_times", controller, grid_size, n_agents)
            for controller in controller_file_names
        ],
    }


def style_boxplot(boxplot, colors, face_mode):
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        if face_mode == "filled":
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        else:
            patch.set_facecolor("white")
            patch.set_alpha(1.0)
            patch.set_hatch("///")

    for whisker, color in zip(boxplot["whiskers"], [color for color in colors for _ in range(2)]):
        whisker.set_color(color)
    for cap, color in zip(boxplot["caps"], [color for color in colors for _ in range(2)]):
        cap.set_color(color)
    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)


def plot_paired_boxplots(ax, grid_label, grid_data, xlabel=False, show_xtick_labels=True):
    distribution_positions = [1, 2]
    offsets = [-0.27, -0.09, 0.09, 0.27]
    task_positions = [distribution_positions[0] + offset for offset in offsets]
    service_positions = [distribution_positions[1] + offset for offset in offsets]

    task_boxplot = ax.boxplot(
        grid_data["task_times"],
        positions=task_positions,
        widths=0.14,
        patch_artist=True,
        manage_ticks=False,
    )
    service_boxplot = ax.boxplot(
        grid_data["service_times"],
        positions=service_positions,
        widths=0.14,
        patch_artist=True,
        manage_ticks=False,
    )

    style_boxplot(task_boxplot, controller_colors, "hatched")
    style_boxplot(service_boxplot, controller_colors, "filled")

    ax.set_ylabel(grid_label, fontweight="bold")
    ax.set_xticks(distribution_positions)
    if show_xtick_labels:
        ax.set_xticklabels(["Task Time", "Service Time"])
    else:
        ax.set_xticklabels([])
    # if xlabel:
    #     ax.set_xlabel("Distribution", fontweight="bold")
    ax.margins(x=0.05)


data_5 = load_grid_data("5", "10")
data_10 = load_grid_data("10", "30")
data_15 = load_grid_data("15", "80")


fig = plt.figure(figsize=(6.0, 8.0))

plot_paired_boxplots(
    plt.subplot(3, 1, 1),
    "5x5 Grid\n(10 agents)",
    data_5,
    show_xtick_labels=False,
)
plot_paired_boxplots(
    plt.subplot(3, 1, 2),
    "10x10 Grid\n(30 agents)",
    data_10,
    show_xtick_labels=False,
)
plot_paired_boxplots(plt.subplot(3, 1, 3), "15x15 Grid\n(80 agents)", data_15, xlabel=True)

legend_handles = [
    Patch(facecolor=color, edgecolor=color, alpha=0.45, label=label)
    for color, label in zip(controller_colors, controller_labels)
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=4,
    fontsize="small",
)

plt.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
plt.savefig("Figure_2.png", dpi=300)
plt.show()
