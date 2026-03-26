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


def get_markdown_table_str(title, data, algorithms, agents):
    # data[algo][agent] -> (mean, std)

    # 1. Gather all data
    header = ["Algorithm"] + [str(a) for a in agents]
    table_rows = []

    for algo in algorithms:
        row = [algo]
        for agent in agents:
            if agent in data.get(algo, {}):
                mean, std = data[algo][agent]
                if title.startswith("Avg") or title.startswith("Std"):
                    val = f"{mean:.2f} ({std:.2f})"
                else:
                    val = f"{mean:.1f} ({std:.1f})"
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

            def summarize(x):
                x = np.array(x, dtype=float)
                return x.mean(), x.std(
                    ddof=0
                )  # population std; use ddof=1 for sample std

            avg_astar, std_astar = summarize(astar_calls_list)
            avg_completed, std_completed = summarize(n_completed_tasks_list)

            avg_total_task, std_total_task = summarize(n_total_task_time_list)
            avg_avg_task, std_avg_task = summarize(n_average_task_time_list)
            avg_std_task, std_std_task = summarize(n_std_task_time_list)

            avg_total_srv, std_total_srv = summarize(n_total_service_time_list)
            avg_avg_srv, std_avg_srv = summarize(n_average_service_time_list)
            avg_std_srv, std_std_srv = summarize(n_std_service_time_list)

            avg_avg_increase, std_avg_increase = summarize(
                n_average_service_increase_list
            )

            # Per Agent Metrics
            # Service Time
            avg_avg_srv_per_agent, std_avg_srv_per_agent = summarize(
                n_avg_service_time_per_agent_list
            )
            # Service Increase
            avg_avg_inc_per_agent, std_avg_inc_per_agent = summarize(
                n_avg_service_increase_per_agent_list
            )

            metrics = {
                "A* Calls": (avg_astar, std_astar),
                "Completed Tasks": (avg_completed, std_completed),
                "Total Task Time (incl. Reallocation) (all agents)": (
                    avg_total_task,
                    std_total_task,
                ),
                "Avg Task Time (incl. Reallocation) (all agents)": (
                    avg_avg_task,
                    std_avg_task,
                ),
                "Std Task Time (incl. Reallocation) (all agents)": (
                    avg_std_task,
                    std_std_task,
                ),
                "Total Service Time (all agents)": (avg_total_srv, std_total_srv),
                "Avg Service Time (all agents)": (avg_avg_srv, std_avg_srv),
                "Std Service Time (all agents)": (avg_std_srv, std_std_srv),
                "Avg Service Time Increase (%) (all agents)": (
                    avg_avg_increase,
                    std_avg_increase,
                ),
                # New Metrics
                "Avg Service Time (per agent mean)": (
                    avg_avg_srv_per_agent,
                    std_avg_srv_per_agent,
                ),
                "Avg Service Increase (%) (per agent mean)": (
                    avg_avg_inc_per_agent,
                    std_avg_inc_per_agent,
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
                "A* calls:                                             mean = {:.3f} \t std = {:.3f}".format(
                    avg_astar, std_astar
                )
            )
            report_lines.append(
                "Completed tasks:                                      mean = {:.3f} \t std = {:.3f}".format(
                    avg_completed, std_completed
                )
            )
            report_lines.append(
                "Total Task Time (incl. Reallocation) (all agents):    mean = {:.3f} \t std = {:.3f}".format(
                    avg_total_task, std_total_task
                )
            )
            report_lines.append(
                "Avg Task Time (incl. Reallocation) (all agents):      mean = {:.3f} \t std = {:.3f}".format(
                    avg_avg_task, std_avg_task
                )
            )
            report_lines.append(
                "Std Task Time (incl. Reallocation) (all agents):      mean = {:.3f} \t std = {:.3f}".format(
                    avg_std_task, std_std_task
                )
            )
            report_lines.append(
                "Total Service Time (all agents):                      mean = {:.3f} \t std = {:.3f}".format(
                    avg_total_srv, std_total_srv
                )
            )
            report_lines.append(
                "Avg Service Time (all agents):                        mean = {:.3f} \t std = {:.3f}".format(
                    avg_avg_srv, std_avg_srv
                )
            )
            report_lines.append(
                "Std Service Time (all agents):                        mean = {:.3f} \t std = {:.3f}".format(
                    avg_std_srv, std_std_srv
                )
            )
            report_lines.append(
                "Avg Service Time Increase (%) (all agents):           mean = {:.3f}% \t std = {:.3f}%".format(
                    avg_avg_increase, std_avg_increase
                )
            )

            report_lines.append(
                "Avg Service Time (per agent mean):                    mean = {:.3f} \t std = {:.3f}".format(
                    avg_avg_srv_per_agent, std_avg_srv_per_agent
                )
            )

            report_lines.append(
                "Avg Service Increase (%) (per agent mean):            mean = {:.3f}% \t std = {:.3f}%".format(
                    avg_avg_inc_per_agent, std_avg_inc_per_agent
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
