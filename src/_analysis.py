"""
POTENTIAL TITLE:
    KARMA MECHANISMS FOR DECENTRALIZED, ORIENTATION-AWARE MAPF

interesting repo: https://github.com/GavinPHR/Multi-Agent-Path-Finding?tab=readme-ov-file
"""

###############################################################################
###### IMPORTS ################################################################
###############################################################################
import numpy as np
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from environment import Environment
from planner_path_astar import AStarPathPlanner

from constants import (
    MAPF_CONTROLLER_CENTRALIZED,
    MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_TRIP_KARMA,
)

# ensure results directory exists
os.makedirs("results", exist_ok=True)


def gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def summarize(x):
    x = np.array(x, dtype=float)
    return (
        x.mean(),
        x.std(ddof=0),
        np.median(x),
        gini(x),
        np.percentile(x, 75) - np.percentile(x, 25),
    )  # population std; use ddof=1 for sample std


def get_markdown_table_str(title, data, algorithms, agents):
    # data[algo][agent] -> (mean, std, median, gini, iqr)

    # 1. Gather all data
    header = ["Algorithm"] + [str(a) for a in agents]
    table_rows = []

    for algo in algorithms:
        row = [algo]
        for agent in agents:
            if agent in data.get(algo, {}):
                mean, std, median, gini_val, iqr = data[algo][agent]
                if title.startswith("Avg") or title.startswith("Std"):
                    val = f"{mean:.2f} ({std:.2f}) | {median:.2f} | {gini_val:.2f} | {iqr:.2f}"
                else:
                    val = f"{mean:.1f} ({std:.1f}) | {median:.1f} | {gini_val:.1f} | {iqr:.1f}"
                row.append(val)
            else:
                row.append("-")
        table_rows.append(row)

    # 2. Determine column widths
    # Initialize with header lengths
    col_widths = [len(h) for h in header]

    # Update with maximum row content lengths
    for row in table_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # 3. Build table string
    lines = []
    lines.append(f"\n{title}")

    # Header
    header_str = (
        "| " + " | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths)) + " |"
    )
    lines.append(header_str)

    # Separator
    separator_str = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines.append(separator_str)

    # Rows
    for row in table_rows:
        row_str = (
            "| " + " | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)) + " |"
        )
        lines.append(row_str)

    return "\n".join(lines)


def print_markdown_table(title, data, algorithms, agents):
    print(get_markdown_table_str(title, data, algorithms, agents))


def plot_metric(
    metric_name,
    data,
    grid_size,
    agents,
    controllers,
    output_dir="results/figs",
):
    """
    Plots a given metric for different algorithms using different plot types.
    """
    # Line plot for mean and std
    _plot_line(metric_name, data, grid_size, agents, controllers, output_dir)
    # Box plot for distribution
    _plot_box(metric_name, data, grid_size, agents, controllers, output_dir)


def _plot_line(
    metric_name, data, grid_size, agents, controllers, output_dir="results/figs"
):
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(12, 8))

    for controller in controllers:
        means = [
            data[grid_size][controller][n_agent][metric_name][0]
            for n_agent in agents
            if n_agent in data[grid_size][controller]
        ]
        stds = [
            data[grid_size][controller][n_agent][metric_name][1]
            for n_agent in agents
            if n_agent in data[grid_size][controller]
        ]

        valid_agents = [
            n_agent for n_agent in agents if n_agent in data[grid_size][controller]
        ]

        ax.plot(
            valid_agents, means, marker="o", linestyle="-", label=f"{controller} (Mean)"
        )
        ax.fill_between(
            valid_agents,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
        )

    ax.set_xlabel("Number of Agents", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_title(
        f"Mean {metric_name} vs. Number of Agents (Grid Size: {grid_size})",
        fontsize=16,
    )
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True)

    filename = f"line_{metric_name.replace(' ', '_').replace('%', 'perc')}_vs_agents_grid_{grid_size}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved plot: {filepath}")


def _plot_box(
    metric_name, data, grid_size, agents, controllers, output_dir="results/figs"
):
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(16, 8))

    plot_data = []
    for n_agent in agents:
        for controller in controllers:
            if (
                "raw_data" in data[grid_size][controller]
                and n_agent in data[grid_size][controller]["raw_data"]
                and metric_name in data[grid_size][controller]["raw_data"][n_agent]
            ):
                raw_values = data[grid_size][controller]["raw_data"][n_agent][
                    metric_name
                ]
                for val in raw_values:
                    plot_data.append(
                        {"Agents": n_agent, "Controller": controller, "Value": val}
                    )

    if not plot_data:
        print(f"No raw data to plot for {metric_name} (box plot).")
        plt.close(fig)
        return

    df = pd.DataFrame(plot_data)

    sns.boxplot(x="Agents", y="Value", hue="Controller", data=df, ax=ax)

    ax.set_xlabel("Number of Agents", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_title(
        f"Distribution of {metric_name} vs. Number of Agents (Grid Size: {grid_size})",
        fontsize=16,
    )
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True)

    filename = f"box_{metric_name.replace(' ', '_').replace('%', 'perc')}_vs_agents_grid_{grid_size}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved plot: {filepath}")


random_seeds = range(41, 51)
grid_sizes = [5]  # , 10, 15, 20]

n_agents_map = {
    5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    10: [1, 5, 10, 15, 20, 25, 30],
    15: [1, 10, 20, 30, 40, 50, 60, 70, 80],
    20: [1, 10, 20, 40, 60, 80, 100, 120, 140],
}
results_summary = {}

# Define base simulation settings
base_simulation_settings = {
    "time_horizon_visualization": 10,
    "time_simulation_duration": 1000,
    "params_astar": {"max_iterations": 5000, "planning_horizon": 50},
    "params_cbs": {
        "max_iterations": 5000,
        "MAX_IDLE_TIME_CONSIDERED": 5,
        "PLANNING_HORIZON": 100,
    },
    "params_karma": {
        "initial_karma": 0,
        "delta_threshold": 1,
        "karma_payment": 1,
        "karma_influence": 0.2,
    },
    "debug_statements": False,
}

# clear summary file
with open("results/summary.txt", "w") as f:
    f.write("Summary of Results\n")

for grid_size in grid_sizes:
    results_summary[grid_size] = {}
    controllers_run = []
    for controller in [
        # MAPF_CONTROLLER_CENTRALIZED,
        MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_TRIP_KARMA,
    ]:
        if controller not in controllers_run:
            controllers_run.append(controller)
            results_summary[grid_size][controller] = {}

        for n_agent in n_agents_map[grid_size]:
            simulation_settings = base_simulation_settings.copy()
            simulation_settings.update(
                {
                    "random_seed": 42,
                    "grid_size": grid_size + 2,
                    "n_agents": n_agent,
                    "mapf_control": controller,
                }
            )

            astar_calls_list = []
            n_completed_tasks_list = []

            # Task Time (incl. Reallocation) metrics (spawn -> complete, including reallocation time)
            n_total_task_time_list = []
            n_average_task_time_list = []
            n_std_task_time_list = []

            # Service Time metrics (pickup -> complete)
            n_total_service_time_list = []
            n_average_service_time_list = []
            n_std_service_time_list = []

            # Service Time Increase Percentage metrics ((service - min) / min)
            n_average_service_increase_list = []
            n_std_service_increase_list = []

            # PER AGENT METRICS
            # Service Time (per agent)
            n_avg_service_time_per_agent_list = []

            # Service Time Increase (per agent)
            n_avg_service_increase_per_agent_list = []

            for random_seed in random_seeds:
                simulation_settings["random_seed"] = random_seed
                n_astar_calls = 0
                if simulation_settings["debug_statements"]:
                    print("Starting Experiment", random_seed)
                environment = Environment(settings=simulation_settings)
                # spawn initial agents
                for n in range(0, simulation_settings["n_agents"]):  # int(N_AGENTS/2)):
                    environment.spawn_agent()
                # spawn initial tasks
                for n in range(0, simulation_settings["n_agents"]):  # int(N_AGENTS/2)):
                    environment.spawn_task()
                # simulation loop
                while (
                    environment.time < environment.settings["time_simulation_duration"]
                ):
                    if simulation_settings["debug_statements"]:
                        print(
                            "\ttime:",
                            environment.time,
                            "\t| agents:",
                            len(environment.agents),
                            "\t| tasks:",
                            len(environment.tasks),
                        )

                    # general update
                    environment.time += 1

                    # handle agents
                    environment.handle_agents()

                    # # spawn tasks randomly
                    while len(environment.tasks) < len(environment.agents):
                        old_len = len(environment.tasks)
                        environment.spawn_task()
                        if old_len == len(
                            environment.tasks
                        ):  # currently too crowded to spawn
                            break

                    # handle tasks
                    environment.assign_open_tasks()
                    closed = environment.close_finished_tasks()

                    # report A-STAR Calls
                    n_astar_calls += AStarPathPlanner.COUNTER
                    AStarPathPlanner.COUNTER = 0

                ############################################################
                # COMPUTE METRICS
                # 0. Flatten tasks from all agents
                all_completed_tasks = [
                    task
                    for task_list in environment.completed_tasks.values()
                    for task in task_list
                ]

                # 1. Task Time (incl. Reallocation) (spawn -> complete)
                task_total_times = [
                    task.completed_time - task.spawned_time
                    for task in all_completed_tasks
                    if task.completed_time is not None
                ]

                # 2. Service Time (pickup -> complete)
                task_service_times = [
                    task.completed_time - task.pickup_time
                    for task in all_completed_tasks
                    if task.completed_time is not None and task.pickup_time is not None
                ]

                # 3. Minimum Task Time
                task_min_times = [
                    task.minimum_task_time
                    for task in all_completed_tasks
                    if task.minimum_task_time != 0
                ]

                # 4. Service Time Increase Percentage
                # (service_time - min_time) / min_time
                task_service_increase_percentages = []
                for s_time, m_time in zip(task_service_times, task_min_times):
                    task_service_increase_percentages.append(
                        (s_time - m_time) / m_time * 100
                    )

                # if the number of tracked tasks is different, raise an error
                if len(task_total_times) != len(task_service_times) or len(
                    task_total_times
                ) != len(task_service_increase_percentages):
                    raise ValueError(
                        "Number of completed tasks with total time does not match number of completed tasks with service time or service time increase percentages."
                    )

                # Task Time (incl. Reallocation) stats
                n_total_task_time = np.sum(task_total_times)
                n_average_task_time = np.mean(task_total_times)
                n_std_task_time = np.std(task_total_times)

                # Service Time stats
                n_total_service_time = np.sum(task_service_times)
                n_average_service_time = np.mean(task_service_times)
                n_std_service_time = np.std(task_service_times)

                # Service Time Increase stats
                n_average_service_increase = np.mean(task_service_increase_percentages)
                n_std_service_increase = np.std(task_service_increase_percentages)

                # 5. Per Agent Metrics
                # For each agent, compute their average service time and average increase
                list_agent_avg_service_times = []
                list_agent_avg_service_increases = []

                for agent_id, agent_tasks in environment.completed_tasks.items():
                    if len(agent_tasks) == 0:
                        continue

                    # Service Times for this agent
                    a_service_times = [
                        task.completed_time - task.pickup_time
                        for task in agent_tasks
                        if task.completed_time is not None
                        and task.pickup_time is not None
                    ]

                    # Service Time Increases for this agent
                    a_min_times = [
                        task.minimum_task_time
                        for task in agent_tasks
                        if task.minimum_task_time != 0
                    ]
                    a_increases = []
                    for s, m in zip(a_service_times, a_min_times):
                        a_increases.append((s - m) / m * 100)

                    if len(a_service_times) > 0:
                        list_agent_avg_service_times.append(np.mean(a_service_times))

                    if len(a_increases) > 0:
                        list_agent_avg_service_increases.append(np.mean(a_increases))

                # Compute stats across agents for this run
                # Mean of agent averages -> "How does the average agent perform?"
                n_avg_service_time_per_agent = (
                    np.mean(list_agent_avg_service_times)
                    if list_agent_avg_service_times
                    else 0
                )

                n_avg_service_increase_per_agent = (
                    np.mean(list_agent_avg_service_increases)
                    if list_agent_avg_service_increases
                    else 0
                )

                # store metrics
                n_completed_tasks = len(task_total_times)
                astar_calls_list.append(n_astar_calls)
                n_completed_tasks_list.append(n_completed_tasks)

                n_total_task_time_list.append(n_total_task_time)
                n_average_task_time_list.append(n_average_task_time)
                n_std_task_time_list.append(n_std_task_time)

                n_total_service_time_list.append(n_total_service_time)
                n_average_service_time_list.append(n_average_service_time)
                n_std_service_time_list.append(n_std_service_time)

                n_average_service_increase_list.append(n_average_service_increase)
                n_std_service_increase_list.append(n_std_service_increase)

                n_avg_service_time_per_agent_list.append(n_avg_service_time_per_agent)

                n_avg_service_increase_per_agent_list.append(
                    n_avg_service_increase_per_agent
                )

            avg_astar, std_astar, median_astar, gini_astar, iqr_astar = summarize(
                astar_calls_list
            )
            (
                avg_completed,
                std_completed,
                median_completed,
                gini_completed,
                iqr_completed,
            ) = summarize(n_completed_tasks_list)

            (
                avg_total_task,
                std_total_task,
                median_total_task,
                gini_total_task,
                iqr_total_task,
            ) = summarize(n_total_task_time_list)
            (
                avg_avg_task,
                std_avg_task,
                median_avg_task,
                gini_avg_task,
                iqr_avg_task,
            ) = summarize(n_average_task_time_list)
            (
                avg_std_task,
                std_std_task,
                median_std_task,
                gini_std_task,
                iqr_std_task,
            ) = summarize(n_std_task_time_list)

            (
                avg_total_srv,
                std_total_srv,
                median_total_srv,
                gini_total_srv,
                iqr_total_srv,
            ) = summarize(n_total_service_time_list)
            (
                avg_avg_srv,
                std_avg_srv,
                median_avg_srv,
                gini_avg_srv,
                iqr_avg_srv,
            ) = summarize(n_average_service_time_list)
            (
                avg_std_srv,
                std_std_srv,
                median_std_srv,
                gini_std_srv,
                iqr_std_srv,
            ) = summarize(n_std_service_time_list)

            (
                avg_avg_increase,
                std_avg_increase,
                median_avg_increase,
                gini_avg_increase,
                iqr_avg_increase,
            ) = summarize(n_average_service_increase_list)

            # Per Agent Metrics
            # Service Time
            (
                avg_avg_srv_per_agent,
                std_avg_srv_per_agent,
                median_avg_srv_per_agent,
                gini_avg_srv_per_agent,
                iqr_avg_srv_per_agent,
            ) = summarize(n_avg_service_time_per_agent_list)
            # Service Increase
            (
                avg_avg_inc_per_agent,
                std_avg_inc_per_agent,
                median_avg_inc_per_agent,
                gini_avg_inc_per_agent,
                iqr_avg_inc_per_agent,
            ) = summarize(n_avg_service_increase_per_agent_list)

            # Store raw data for box plots
            if "raw_data" not in results_summary[grid_size][controller]:
                results_summary[grid_size][controller]["raw_data"] = {}
            if n_agent not in results_summary[grid_size][controller]["raw_data"]:
                results_summary[grid_size][controller]["raw_data"][n_agent] = {}

            raw_data_storage = results_summary[grid_size][controller]["raw_data"][
                n_agent
            ]
            raw_data_storage["A* Calls"] = astar_calls_list
            raw_data_storage["Completed Tasks"] = n_completed_tasks_list
            raw_data_storage["Total Task Time (incl. Reallocation) (all agents)"] = (
                n_total_task_time_list
            )
            raw_data_storage["Avg Task Time (incl. Reallocation) (all agents)"] = (
                n_average_task_time_list
            )
            raw_data_storage["Std Task Time (incl. Reallocation) (all agents)"] = (
                n_std_task_time_list
            )
            raw_data_storage["Total Service Time (all agents)"] = (
                n_total_service_time_list
            )
            raw_data_storage["Avg Service Time (all agents)"] = (
                n_average_service_time_list
            )
            raw_data_storage["Std Service Time (all agents)"] = n_std_service_time_list
            raw_data_storage["Avg Service Time Increase (%) (all agents)"] = (
                n_average_service_increase_list
            )
            raw_data_storage["Avg Service Time (per agent mean)"] = (
                n_avg_service_time_per_agent_list
            )
            raw_data_storage["Avg Service Increase (%) (per agent mean)"] = (
                n_avg_service_increase_per_agent_list
            )

            metrics = {
                "A* Calls": (
                    avg_astar,
                    std_astar,
                    median_astar,
                    gini_astar,
                    iqr_astar,
                ),
                "Completed Tasks": (
                    avg_completed,
                    std_completed,
                    median_completed,
                    gini_completed,
                    iqr_completed,
                ),
                "Total Task Time (incl. Reallocation) (all agents)": (
                    avg_total_task,
                    std_total_task,
                    median_total_task,
                    gini_total_task,
                    iqr_total_task,
                ),
                "Avg Task Time (incl. Reallocation) (all agents)": (
                    avg_avg_task,
                    std_avg_task,
                    median_avg_task,
                    gini_avg_task,
                    iqr_avg_task,
                ),
                "Std Task Time (incl. Reallocation) (all agents)": (
                    avg_std_task,
                    std_std_task,
                    median_std_task,
                    gini_std_task,
                    iqr_std_task,
                ),
                "Total Service Time (all agents)": (
                    avg_total_srv,
                    std_total_srv,
                    median_total_srv,
                    gini_total_srv,
                    iqr_total_srv,
                ),
                "Avg Service Time (all agents)": (
                    avg_avg_srv,
                    std_avg_srv,
                    median_avg_srv,
                    gini_avg_srv,
                    iqr_avg_srv,
                ),
                "Std Service Time (all agents)": (
                    avg_std_srv,
                    std_std_srv,
                    median_std_srv,
                    gini_std_srv,
                    iqr_std_srv,
                ),
                "Avg Service Time Increase (%) (all agents)": (
                    avg_avg_increase,
                    std_avg_increase,
                    median_avg_increase,
                    gini_avg_increase,
                    iqr_avg_increase,
                ),
                # New Metrics
                "Avg Service Time (per agent mean)": (
                    avg_avg_srv_per_agent,
                    std_avg_srv_per_agent,
                    median_avg_srv_per_agent,
                    gini_avg_srv_per_agent,
                    iqr_avg_srv_per_agent,
                ),
                "Avg Service Increase (%) (per agent mean)": (
                    avg_avg_inc_per_agent,
                    std_avg_inc_per_agent,
                    median_avg_inc_per_agent,
                    gini_avg_inc_per_agent,
                    iqr_avg_inc_per_agent,
                ),
            }
            results_summary[grid_size][controller][n_agent] = metrics

            report_lines = []
            report_lines.append("=====================================================")
            report_lines.append(
                "Experiment Results ["
                + str(simulation_settings["n_agents"])
                + " agents] ["
                + str(simulation_settings["grid_size"] - 2)
                + " grid-size] for algorithm "
                + str(simulation_settings["mapf_control"])
                + " over "
                + str(len(astar_calls_list))
                + " experiments"
            )
            report_lines.append("=====================================================")
            report_lines.append(
                "A* calls:                                             mean = {:.3f} \t std = {:.3f} \t median = {:.3f} \t gini = {:.3f} \t iqr = {:.3f}".format(
                    avg_astar, std_astar, median_astar, gini_astar, iqr_astar
                )
            )
            report_lines.append(
                "Completed tasks:                                      mean = {:.3f} \t std = {:.3f} \t median = {:.3f} \t gini = {:.3f} \t iqr = {:.3f}".format(
                    avg_completed,
                    std_completed,
                    median_completed,
                    gini_completed,
                    iqr_completed,
                )
            )
            report_lines.append(
                "Total Task Time (incl. Reallocation) (all agents):    mean = {:.3f} \t std = {:.3f} \t median = {:.3f} \t gini = {:.3f} \t iqr = {:.3f}".format(
                    avg_total_task,
                    std_total_task,
                    median_total_task,
                    gini_total_task,
                    iqr_total_task,
                )
            )
            report_lines.append(
                "Avg Task Time (incl. Reallocation) (all agents):      mean = {:.3f} \t std = {:.3f} \t median = {:.3f} \t gini = {:.3f} \t iqr = {:.3f}".format(
                    avg_avg_task,
                    std_avg_task,
                    median_avg_task,
                    gini_avg_task,
                    iqr_avg_task,
                )
            )
            report_lines.append(
                "Std Task Time (incl. Reallocation) (all agents):      mean = {:.3f} \t std = {:.3f} \t median = {:.3f} \t gini = {:.3f} \t iqr = {:.3f}".format(
                    avg_std_task,
                    std_std_task,
                    median_std_task,
                    gini_std_task,
                    iqr_std_task,
                )
            )
            report_lines.append(
                "Total Service Time (all agents):                      mean = {:.3f} \t std = {:.3f} \t median = {:.3f} \t gini = {:.3f} \t iqr = {:.3f}".format(
                    avg_total_srv,
                    std_total_srv,
                    median_total_srv,
                    gini_total_srv,
                    iqr_total_srv,
                )
            )
            report_lines.append(
                "Avg Service Time (all agents):                        mean = {:.3f} \t std = {:.3f} \t median = {:.3f} \t gini = {:.3f} \t iqr = {:.3f}".format(
                    avg_avg_srv, std_avg_srv, median_avg_srv, gini_avg_srv, iqr_avg_srv
                )
            )
            report_lines.append(
                "Std Service Time (all agents):                        mean = {:.3f} \t std = {:.3f} \t median = {:.3f} \t gini = {:.3f} \t iqr = {:.3f}".format(
                    avg_std_srv, std_std_srv, median_std_srv, gini_std_srv, iqr_std_srv
                )
            )
            report_lines.append(
                "Avg Service Time Increase (%) (all agents):           mean = {:.3f}% \t std = {:.3f}% \t median = {:.3f}% \t gini = {:.3f} \t iqr = {:.3f}%".format(
                    avg_avg_increase,
                    std_avg_increase,
                    median_avg_increase,
                    gini_avg_increase,
                    iqr_avg_increase,
                )
            )

            report_lines.append(
                "Avg Service Time (per agent mean):                    mean = {:.3f} \t std = {:.3f} \t median = {:.3f} \t gini = {:.3f} \t iqr = {:.3f}".format(
                    avg_avg_srv_per_agent,
                    std_avg_srv_per_agent,
                    median_avg_srv_per_agent,
                    gini_avg_srv_per_agent,
                    iqr_avg_srv_per_agent,
                )
            )

            report_lines.append(
                "Avg Service Increase (%) (per agent mean):            mean = {:.3f}% \t std = {:.3f}% \t median = {:.3f}% \t gini = {:.3f} \t iqr = {:.3f}%".format(
                    avg_avg_inc_per_agent,
                    std_avg_inc_per_agent,
                    median_avg_inc_per_agent,
                    gini_avg_inc_per_agent,
                    iqr_avg_inc_per_agent,
                )
            )

            report_lines.append(
                "=====================================================\n"
            )

            report_str = "\n".join(report_lines)
            print(report_str)

            # Store result in txt file
            # If it is the first agent config, write over, else append
            mode = "w" if n_agent == n_agents_map[grid_size][0] else "a"
            with open(f"results/results_{controller}_{grid_size}.txt", mode) as f:
                f.write(report_str + "\n")

    # Print tables for this grid_size
    agents = n_agents_map[grid_size]

    with open("results/summary.txt", "a") as f_summary:
        f_summary.write(f"\n\n### GRID SIZE {grid_size} SUMMARY ###\n")

        # Write simulation settings
        f_summary.write("Simulation Settings:\n")
        for key, value in base_simulation_settings.items():
            f_summary.write(f"  {key}: {value}\n")
        f_summary.write(f"  random_seeds_range: {random_seeds}\n")
        f_summary.write("\n")

        for metric_name in [
            "A* Calls",
            "Completed Tasks",
            "Total Task Time (incl. Reallocation) (all agents)",
            "Avg Task Time (incl. Reallocation) (all agents)",
            "Std Task Time (incl. Reallocation) (all agents)",
            "Total Service Time (all agents)",
            "Avg Service Time (all agents)",
            "Std Service Time (all agents)",
            "Avg Service Time Increase (%) (all agents)",
            "Avg Service Time (per agent mean)",
            "Avg Service Increase (%) (per agent mean)",
        ]:
            metric_data = {}
            for algo in controllers_run:
                metric_data[algo] = {}
                for agent in agents:
                    if agent in results_summary.get(grid_size, {}).get(algo, {}):
                        metric_data[algo][agent] = results_summary[grid_size][algo][
                            agent
                        ][metric_name]

            table_str = get_markdown_table_str(
                metric_name, metric_data, controllers_run, agents
            )
            print(table_str)
            f_summary.write(table_str + "\n")

    # Save the results_summary to a JSON file for plotting
    # Convert keys to strings for JSON compatibility
    results_summary_str_keys = {}
    for k, v in results_summary.items():
        grid_size_key = str(k)
        results_summary_str_keys[grid_size_key] = {}
        for controller, controller_data in v.items():
            controller_key = str(controller)
            results_summary_str_keys[grid_size_key][controller_key] = {}
            for n_agent, agent_data in controller_data.items():
                agent_key = str(n_agent)
                results_summary_str_keys[grid_size_key][controller_key][
                    agent_key
                ] = agent_data

    with open("results/summary.json", "w") as f_json:
        # Custom encoder to handle numpy arrays
        class NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                return json.JSONEncoder.default(self, o)

        json.dump(results_summary_str_keys, f_json, indent=4, cls=NumpyEncoder)

    # Plotting results
    for grid_size_str, grid_data in results_summary_str_keys.items():
        grid_size = int(grid_size_str)
        controllers = [c for c in grid_data.keys() if c != "raw_data"]

        # Determine the union of agents for this grid size
        all_agents = set()
        for controller in controllers:
            all_agents.update(grid_data[controller].keys())

        # Convert agent keys from string to int and sort
        agents = sorted([int(a) for a in all_agents if a.isdigit()])

        # Get all metric names from the first available data point
        metric_names = []
        if controllers and agents:
            first_controller = controllers[0]
            first_agent_str = str(agents[0])
            if first_agent_str in grid_data[first_controller]:
                metric_names = [
                    k
                    for k in grid_data[first_controller][first_agent_str].keys()
                    if k != "raw_data"
                ]

        if not metric_names:
            print(f"No metrics found for grid size {grid_size}. Skipping.")
            continue

        for metric_name in metric_names:
            plot_metric(
                metric_name,
                results_summary,
                grid_size,
                agents,
                controllers,
            )
