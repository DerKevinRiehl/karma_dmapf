"""
POTENTIAL TITLE:
    KARMA MECHANISMS FOR DECENTRALIZED, ORIENTATION-AWARE MAPF

interesting repo: https://github.com/GavinPHR/Multi-Agent-Path-Finding?tab=readme-ov-file
"""

###############################################################################
###### IMPORTS ################################################################
###############################################################################
import numpy as np
from environment import Environment
from planner_path_astar import AStarPathPlanner

from constants import (
    MAPF_CONTROLLER_CENTRALIZED,
    MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA,
)


def print_markdown_table(title, data, algorithms, agents):
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

    # 3. Print table with formatting
    print(f"\n{title}")

    # Header
    header_str = (
        "| " + " | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths)) + " |"
    )
    print(header_str)

    # Separator
    separator_str = "| " + " | ".join("-" * w for w in col_widths) + " |"
    print(separator_str)

    # Rows
    for row in table_rows:
        row_str = (
            "| " + " | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)) + " |"
        )
        print(row_str)


random_seeds = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

grid_sizes = [5]  # , 10, 15, 20]

n_agents_map = {
    5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    15: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    20: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
results_summary = {}

for grid_size in grid_sizes:
    results_summary[grid_size] = {}
    controllers_run = []
    for controller in [
        # MAPF_CONTROLLER_CENTRALIZED,
        MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA,
    ]:
        if controller not in controllers_run:
            controllers_run.append(controller)
            results_summary[grid_size][controller] = {}

        for n_agent in n_agents_map[grid_size]:
            simulation_settings = {
                "random_seed": 42,
                "grid_size": grid_size + 2,
                "n_agents": n_agent,
                "mapf_control": controller,
                "time_horizon_visualization": 10,
                "time_simulation_duration": 1000,
                "params_astar": {"max_iterations": 5000, "planning_horizon": 50},
                "params_cbs": {
                    "max_iterations": 5000,
                    "MAX_IDLE_TIME_CONSIDERED": 5,
                    "PLANNING_HORIZON": 100,
                },
                "params_karma": {
                    "initial_karma": 10,
                    "delta_threshold": 1,
                    "karma_payment": 1,
                },
                "debug_statements": False,
            }

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
                # 1. Task Time (incl. Reallocation) (spawn -> complete)
                task_total_times = [
                    task.completed_time - task.spawned_time
                    for task in environment.completed_tasks
                    if task.completed_time is not None
                ]

                # 2. Service Time (pickup -> complete)
                task_service_times = [
                    task.completed_time - task.pickup_time
                    for task in environment.completed_tasks
                    if task.completed_time is not None and task.pickup_time is not None
                ]

                # 3. Minimum Task Time
                task_min_times = [
                    task.minimum_task_time
                    for task in environment.completed_tasks
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

            metrics = {
                "A* Calls": (avg_astar, std_astar),
                "Completed Tasks": (avg_completed, std_completed),
                "Total Task Time (incl. Reallocation)": (
                    avg_total_task,
                    std_total_task,
                ),
                "Avg Task Time (incl. Reallocation)": (avg_avg_task, std_avg_task),
                "Std Task Time (incl. Reallocation)": (avg_std_task, std_std_task),
                "Total Service Time": (avg_total_srv, std_total_srv),
                "Avg Service Time": (avg_avg_srv, std_avg_srv),
                "Std Service Time": (avg_std_srv, std_std_srv),
                "Avg Service Time Increase (%)": (avg_avg_increase, std_avg_increase),
            }
            results_summary[grid_size][controller][n_agent] = metrics

            print("=====================================================")
            print(
                "Experiment Results ["
                + str(simulation_settings["n_agents"])
                + " agents] ["
                + str(simulation_settings["grid_size"] - 2)
                + " grid-size] for algorithm",
                simulation_settings["mapf_control"],
                "over",
                len(astar_calls_list),
                "experiments",
            )
            print("=====================================================")
            print(
                "A* calls:                                mean = {:.3f} \t std = {:.3f}".format(
                    avg_astar, std_astar
                )
            )
            print(
                "Completed tasks:                         mean = {:.3f} \t std = {:.3f}".format(
                    avg_completed, std_completed
                )
            )
            print(
                "Total Task Time (incl. Reallocation):    mean = {:.3f} \t std = {:.3f}".format(
                    avg_total_task, std_total_task
                )
            )
            print(
                "Avg Task Time (incl. Reallocation):      mean = {:.3f} \t std = {:.3f}".format(
                    avg_avg_task, std_avg_task
                )
            )
            print(
                "Std Task Time (incl. Reallocation):      mean = {:.3f} \t std = {:.3f}".format(
                    avg_std_task, std_std_task
                )
            )
            print(
                "Total Service Time:                      mean = {:.3f} \t std = {:.3f}".format(
                    avg_total_srv, std_total_srv
                )
            )
            print(
                "Avg Service Time:                        mean = {:.3f} \t std = {:.3f}".format(
                    avg_avg_srv, std_avg_srv
                )
            )
            print(
                "Std Service Time:                        mean = {:.3f} \t std = {:.3f}".format(
                    avg_std_srv, std_std_srv
                )
            )
            print(
                "Avg Service Time Increase (%):           mean = {:.3f}% \t std = {:.3f}%".format(
                    avg_avg_increase, std_avg_increase
                )
            )
            print("=====================================================\n")

    # Print tables for this grid_size
    agents = n_agents_map[grid_size]

    for metric_name in [
        "A* Calls",
        "Completed Tasks",
        "Total Task Time (incl. Reallocation)",
        "Avg Task Time (incl. Reallocation)",
        "Std Task Time (incl. Reallocation)",
        "Total Service Time",
        "Avg Service Time",
        "Std Service Time",
        "Avg Service Time Increase (%)",
    ]:
        metric_data = {}
        for algo in controllers_run:
            metric_data[algo] = {}
            for agent in agents:
                if agent in results_summary.get(grid_size, {}).get(algo, {}):
                    metric_data[algo][agent] = results_summary[grid_size][algo][agent][
                        metric_name
                    ]

        print_markdown_table(metric_name, metric_data, controllers_run, agents)

"""
=====================================================
Experiment Results [1 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 3627.200 	 std = 95.272
Completed tasks:                         mean = 85.200 	 std = 2.600
Total Task Time (incl. Reallocation):    mean = 910.500 	 std = 4.455
Avg Task Time (incl. Reallocation):      mean = 10.696 	 std = 0.327
Std Task Time (incl. Reallocation):      mean = 2.852 	 std = 0.176
Total Service Time:                      mean = 403.300 	 std = 11.091
Avg Service Time:                        mean = 4.739 	 std = 0.209
Std Service Time:                        mean = 1.933 	 std = 0.105
Avg Service Time Increase (%):           mean = 0.000% 	 std = 0.000%
=====================================================

=====================================================
Experiment Results [2 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 8616.400 	 std = 398.214
Completed tasks:                         mean = 166.500 	 std = 3.775
Total Task Time (incl. Reallocation):    mean = 1822.000 	 std = 7.887
Avg Task Time (incl. Reallocation):      mean = 10.949 	 std = 0.266
Std Task Time (incl. Reallocation):      mean = 2.814 	 std = 0.122
Total Service Time:                      mean = 835.200 	 std = 21.679
Avg Service Time:                        mean = 5.020 	 std = 0.207
Std Service Time:                        mean = 2.021 	 std = 0.118
Avg Service Time Increase (%):           mean = 3.490% 	 std = 1.201%
=====================================================

=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 14903.600 	 std = 515.064
Completed tasks:                         mean = 247.700 	 std = 2.968
Total Task Time (incl. Reallocation):    mean = 2737.000 	 std = 6.033
Avg Task Time (incl. Reallocation):      mean = 11.051 	 std = 0.151
Std Task Time (incl. Reallocation):      mean = 3.048 	 std = 0.078
Total Service Time:                      mean = 1288.300 	 std = 22.791
Avg Service Time:                        mean = 5.201 	 std = 0.075
Std Service Time:                        mean = 2.101 	 std = 0.106
Avg Service Time Increase (%):           mean = 7.460% 	 std = 1.520%
=====================================================

=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 23603.700 	 std = 1281.768
Completed tasks:                         mean = 325.400 	 std = 6.070
Total Task Time (incl. Reallocation):    mean = 3657.400 	 std = 7.338
Avg Task Time (incl. Reallocation):      mean = 11.244 	 std = 0.223
Std Task Time (incl. Reallocation):      mean = 3.214 	 std = 0.113
Total Service Time:                      mean = 1742.100 	 std = 22.762
Avg Service Time:                        mean = 5.355 	 std = 0.113
Std Service Time:                        mean = 2.261 	 std = 0.085
Avg Service Time Increase (%):           mean = 11.757% 	 std = 0.831%
=====================================================

=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 33813.400 	 std = 2082.818
Completed tasks:                         mean = 401.900 	 std = 4.182
Total Task Time (incl. Reallocation):    mean = 4575.600 	 std = 6.070
Avg Task Time (incl. Reallocation):      mean = 11.386 	 std = 0.120
Std Task Time (incl. Reallocation):      mean = 3.370 	 std = 0.080
Total Service Time:                      mean = 2223.300 	 std = 26.207
Avg Service Time:                        mean = 5.532 	 std = 0.070
Std Service Time:                        mean = 2.383 	 std = 0.055
Avg Service Time Increase (%):           mean = 15.360% 	 std = 0.673%
=====================================================

=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 49391.700 	 std = 2446.961
Completed tasks:                         mean = 472.900 	 std = 6.534
Total Task Time (incl. Reallocation):    mean = 5496.500 	 std = 7.420
Avg Task Time (incl. Reallocation):      mean = 11.625 	 std = 0.167
Std Task Time (incl. Reallocation):      mean = 3.477 	 std = 0.118
Total Service Time:                      mean = 2726.100 	 std = 31.156
Avg Service Time:                        mean = 5.766 	 std = 0.109
Std Service Time:                        mean = 2.550 	 std = 0.066
Avg Service Time Increase (%):           mean = 21.120% 	 std = 1.187%
=====================================================

=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 71365.000 	 std = 5322.212
Completed tasks:                         mean = 538.500 	 std = 7.826
Total Task Time (incl. Reallocation):    mean = 6422.700 	 std = 16.769
Avg Task Time (incl. Reallocation):      mean = 11.930 	 std = 0.187
Std Task Time (incl. Reallocation):      mean = 3.729 	 std = 0.114
Total Service Time:                      mean = 3260.800 	 std = 32.875
Avg Service Time:                        mean = 6.057 	 std = 0.118
Std Service Time:                        mean = 2.785 	 std = 0.095
Avg Service Time Increase (%):           mean = 26.554% 	 std = 1.656%
=====================================================

=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 93859.800 	 std = 4993.255
Completed tasks:                         mean = 604.100 	 std = 9.669
Total Task Time (incl. Reallocation):    mean = 7340.700 	 std = 11.172
Avg Task Time (incl. Reallocation):      mean = 12.155 	 std = 0.192
Std Task Time (incl. Reallocation):      mean = 3.942 	 std = 0.113
Total Service Time:                      mean = 3762.700 	 std = 54.325
Avg Service Time:                        mean = 6.230 	 std = 0.124
Std Service Time:                        mean = 2.971 	 std = 0.076
Avg Service Time Increase (%):           mean = 31.508% 	 std = 1.956%
=====================================================

=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 137436.500 	 std = 14003.956
Completed tasks:                         mean = 654.600 	 std = 8.969
Total Task Time (incl. Reallocation):    mean = 8294.400 	 std = 18.688
Avg Task Time (incl. Reallocation):      mean = 12.673 	 std = 0.185
Std Task Time (incl. Reallocation):      mean = 4.264 	 std = 0.097
Total Service Time:                      mean = 4340.700 	 std = 35.288
Avg Service Time:                        mean = 6.632 	 std = 0.111
Std Service Time:                        mean = 3.200 	 std = 0.090
Avg Service Time Increase (%):           mean = 37.801% 	 std = 1.224%
=====================================================

=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:                                mean = 192568.900 	 std = 17932.290
Completed tasks:                         mean = 701.700 	 std = 7.849
Total Task Time (incl. Reallocation):    mean = 9152.200 	 std = 23.613
Avg Task Time (incl. Reallocation):      mean = 13.045 	 std = 0.170
Std Task Time (incl. Reallocation):      mean = 4.456 	 std = 0.099
Total Service Time:                      mean = 4919.400 	 std = 28.524
Avg Service Time:                        mean = 7.011 	 std = 0.081
Std Service Time:                        mean = 3.465 	 std = 0.059
Avg Service Time Increase (%):           mean = 44.596% 	 std = 1.424%
=====================================================

=====================================================
Experiment Results [1 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 3627.200 	 std = 95.272
Completed tasks:                         mean = 85.200 	 std = 2.600
Total Task Time (incl. Reallocation):    mean = 910.500 	 std = 4.455
Avg Task Time (incl. Reallocation):      mean = 10.696 	 std = 0.327
Std Task Time (incl. Reallocation):      mean = 2.852 	 std = 0.176
Total Service Time:                      mean = 403.300 	 std = 11.091
Avg Service Time:                        mean = 4.739 	 std = 0.209
Std Service Time:                        mean = 1.933 	 std = 0.105
Avg Service Time Increase (%):           mean = 0.000% 	 std = 0.000%
=====================================================

=====================================================
Experiment Results [2 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 12704.400 	 std = 829.455
Completed tasks:                         mean = 168.700 	 std = 3.318
Total Task Time (incl. Reallocation):    mean = 1822.200 	 std = 6.210
Avg Task Time (incl. Reallocation):      mean = 10.806 	 std = 0.245
Std Task Time (incl. Reallocation):      mean = 2.692 	 std = 0.151
Total Service Time:                      mean = 836.000 	 std = 21.698
Avg Service Time:                        mean = 4.959 	 std = 0.191
Std Service Time:                        mean = 1.935 	 std = 0.074
Avg Service Time Increase (%):           mean = 1.413% 	 std = 0.719%
=====================================================

=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 28355.700 	 std = 2865.048
Completed tasks:                         mean = 252.400 	 std = 3.555
Total Task Time (incl. Reallocation):    mean = 2732.300 	 std = 6.827
Avg Task Time (incl. Reallocation):      mean = 10.828 	 std = 0.163
Std Task Time (incl. Reallocation):      mean = 2.922 	 std = 0.115
Total Service Time:                      mean = 1262.900 	 std = 14.124
Avg Service Time:                        mean = 5.005 	 std = 0.089
Std Service Time:                        mean = 2.004 	 std = 0.104
Avg Service Time Increase (%):           mean = 3.214% 	 std = 0.638%
=====================================================

=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 56836.800 	 std = 11385.130
Completed tasks:                         mean = 340.300 	 std = 5.292
Total Task Time (incl. Reallocation):    mean = 3641.200 	 std = 11.677
Avg Task Time (incl. Reallocation):      mean = 10.703 	 std = 0.200
Std Task Time (incl. Reallocation):      mean = 2.928 	 std = 0.104
Total Service Time:                      mean = 1700.500 	 std = 21.068
Avg Service Time:                        mean = 4.998 	 std = 0.105
Std Service Time:                        mean = 2.044 	 std = 0.098
Avg Service Time Increase (%):           mean = 4.835% 	 std = 0.641%
=====================================================

=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 96837.100 	 std = 11593.635
Completed tasks:                         mean = 418.800 	 std = 5.582
Total Task Time (incl. Reallocation):    mean = 4559.500 	 std = 7.242
Avg Task Time (incl. Reallocation):      mean = 10.889 	 std = 0.154
Std Task Time (incl. Reallocation):      mean = 3.043 	 std = 0.088
Total Service Time:                      mean = 2147.600 	 std = 24.650
Avg Service Time:                        mean = 5.129 	 std = 0.099
Std Service Time:                        mean = 2.072 	 std = 0.045
Avg Service Time Increase (%):           mean = 6.981% 	 std = 0.862%
=====================================================

=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 174789.300 	 std = 32285.563
Completed tasks:                         mean = 499.300 	 std = 5.640
Total Task Time (incl. Reallocation):    mean = 5470.800 	 std = 11.746
Avg Task Time (incl. Reallocation):      mean = 10.958 	 std = 0.137
Std Task Time (incl. Reallocation):      mean = 3.098 	 std = 0.116
Total Service Time:                      mean = 2596.900 	 std = 15.921
Avg Service Time:                        mean = 5.202 	 std = 0.064
Std Service Time:                        mean = 2.153 	 std = 0.048
Avg Service Time Increase (%):           mean = 9.409% 	 std = 1.450%
=====================================================

=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 271263.200 	 std = 40784.480
Completed tasks:                         mean = 581.300 	 std = 5.178
Total Task Time (incl. Reallocation):    mean = 6383.800 	 std = 10.255
Avg Task Time (incl. Reallocation):      mean = 10.983 	 std = 0.104
Std Task Time (incl. Reallocation):      mean = 3.175 	 std = 0.086
Total Service Time:                      mean = 3070.300 	 std = 23.808
Avg Service Time:                        mean = 5.282 	 std = 0.077
Std Service Time:                        mean = 2.266 	 std = 0.075
Avg Service Time Increase (%):           mean = 12.046% 	 std = 0.811%
=====================================================

=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 487585.600 	 std = 49257.502
Completed tasks:                         mean = 656.500 	 std = 8.801
Total Task Time (incl. Reallocation):    mean = 7302.200 	 std = 11.712
Avg Task Time (incl. Reallocation):      mean = 11.125 	 std = 0.159
Std Task Time (incl. Reallocation):      mean = 3.274 	 std = 0.100
Total Service Time:                      mean = 3540.400 	 std = 27.369
Avg Service Time:                        mean = 5.394 	 std = 0.110
Std Service Time:                        mean = 2.329 	 std = 0.069
Avg Service Time Increase (%):           mean = 14.691% 	 std = 1.451%
=====================================================

=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 743421.200 	 std = 55166.601
Completed tasks:                         mean = 725.300 	 std = 8.427
Total Task Time (incl. Reallocation):    mean = 8227.600 	 std = 14.800
Avg Task Time (incl. Reallocation):      mean = 11.345 	 std = 0.147
Std Task Time (incl. Reallocation):      mean = 3.489 	 std = 0.063
Total Service Time:                      mean = 4047.700 	 std = 37.092
Avg Service Time:                        mean = 5.582 	 std = 0.088
Std Service Time:                        mean = 2.474 	 std = 0.038
Avg Service Time Increase (%):           mean = 17.664% 	 std = 1.231%
=====================================================

=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:                                mean = 1183481.500 	 std = 122521.154
Completed tasks:                         mean = 783.800 	 std = 7.494
Total Task Time (incl. Reallocation):    mean = 9065.800 	 std = 10.600
Avg Task Time (incl. Reallocation):      mean = 11.568 	 std = 0.112
Std Task Time (incl. Reallocation):      mean = 3.625 	 std = 0.123
Total Service Time:                      mean = 4450.600 	 std = 51.247
Avg Service Time:                        mean = 5.679 	 std = 0.078
Std Service Time:                        mean = 2.591 	 std = 0.067
Avg Service Time Increase (%):           mean = 21.019% 	 std = 1.146%
=====================================================

=====================================================
Experiment Results [1 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 3627.200 	 std = 95.272
Completed tasks:                         mean = 85.200 	 std = 2.600
Total Task Time (incl. Reallocation):    mean = 910.500 	 std = 4.455
Avg Task Time (incl. Reallocation):      mean = 10.696 	 std = 0.327
Std Task Time (incl. Reallocation):      mean = 2.852 	 std = 0.176
Total Service Time:                      mean = 403.300 	 std = 11.091
Avg Service Time:                        mean = 4.739 	 std = 0.209
Std Service Time:                        mean = 1.933 	 std = 0.105
Avg Service Time Increase (%):           mean = 0.000% 	 std = 0.000%
=====================================================

=====================================================
Experiment Results [2 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 11362.400 	 std = 614.417
Completed tasks:                         mean = 170.800 	 std = 3.092
Total Task Time (incl. Reallocation):    mean = 1820.200 	 std = 5.381
Avg Task Time (incl. Reallocation):      mean = 10.661 	 std = 0.211
Std Task Time (incl. Reallocation):      mean = 2.816 	 std = 0.189
Total Service Time:                      mean = 820.200 	 std = 14.607
Avg Service Time:                        mean = 4.804 	 std = 0.141
Std Service Time:                        mean = 1.936 	 std = 0.148
Avg Service Time Increase (%):           mean = 0.858% 	 std = 0.260%
=====================================================

=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 25519.700 	 std = 1942.669
Completed tasks:                         mean = 256.300 	 std = 5.367
Total Task Time (incl. Reallocation):    mean = 2726.400 	 std = 9.276
Avg Task Time (incl. Reallocation):      mean = 10.643 	 std = 0.249
Std Task Time (incl. Reallocation):      mean = 2.843 	 std = 0.098
Total Service Time:                      mean = 1253.900 	 std = 19.614
Avg Service Time:                        mean = 4.894 	 std = 0.122
Std Service Time:                        mean = 1.970 	 std = 0.072
Avg Service Time Increase (%):           mean = 2.097% 	 std = 0.474%
=====================================================

=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 48669.700 	 std = 5586.435
Completed tasks:                         mean = 342.100 	 std = 5.009
Total Task Time (incl. Reallocation):    mean = 3639.900 	 std = 7.687
Avg Task Time (incl. Reallocation):      mean = 10.642 	 std = 0.171
Std Task Time (incl. Reallocation):      mean = 2.898 	 std = 0.127
Total Service Time:                      mean = 1695.100 	 std = 34.512
Avg Service Time:                        mean = 4.956 	 std = 0.124
Std Service Time:                        mean = 2.015 	 std = 0.082
Avg Service Time Increase (%):           mean = 3.431% 	 std = 0.625%
=====================================================

=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 80443.600 	 std = 9141.000
Completed tasks:                         mean = 425.300 	 std = 4.124
Total Task Time (incl. Reallocation):    mean = 4551.700 	 std = 10.946
Avg Task Time (incl. Reallocation):      mean = 10.703 	 std = 0.118
Std Task Time (incl. Reallocation):      mean = 2.970 	 std = 0.095
Total Service Time:                      mean = 2164.300 	 std = 25.068
Avg Service Time:                        mean = 5.089 	 std = 0.048
Std Service Time:                        mean = 2.138 	 std = 0.055
Avg Service Time Increase (%):           mean = 5.170% 	 std = 0.467%
=====================================================

=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 127094.900 	 std = 12860.332
Completed tasks:                         mean = 510.000 	 std = 5.138
Total Task Time (incl. Reallocation):    mean = 5460.800 	 std = 9.673
Avg Task Time (incl. Reallocation):      mean = 10.709 	 std = 0.113
Std Task Time (incl. Reallocation):      mean = 3.037 	 std = 0.042
Total Service Time:                      mean = 2629.000 	 std = 31.887
Avg Service Time:                        mean = 5.156 	 std = 0.097
Std Service Time:                        mean = 2.173 	 std = 0.047
Avg Service Time Increase (%):           mean = 6.634% 	 std = 0.799%
=====================================================

=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 206782.500 	 std = 17366.528
Completed tasks:                         mean = 591.700 	 std = 8.125
Total Task Time (incl. Reallocation):    mean = 6375.700 	 std = 13.957
Avg Task Time (incl. Reallocation):      mean = 10.777 	 std = 0.152
Std Task Time (incl. Reallocation):      mean = 3.118 	 std = 0.111
Total Service Time:                      mean = 3074.700 	 std = 33.761
Avg Service Time:                        mean = 5.197 	 std = 0.096
Std Service Time:                        mean = 2.275 	 std = 0.056
Avg Service Time Increase (%):           mean = 8.956% 	 std = 0.756%
=====================================================

=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 324765.900 	 std = 35047.142
Completed tasks:                         mean = 669.300 	 std = 8.855
Total Task Time (incl. Reallocation):    mean = 7286.800 	 std = 19.167
Avg Task Time (incl. Reallocation):      mean = 10.889 	 std = 0.166
Std Task Time (incl. Reallocation):      mean = 3.297 	 std = 0.117
Total Service Time:                      mean = 3563.700 	 std = 52.593
Avg Service Time:                        mean = 5.326 	 std = 0.129
Std Service Time:                        mean = 2.352 	 std = 0.084
Avg Service Time Increase (%):           mean = 11.024% 	 std = 0.967%
=====================================================

=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 491146.600 	 std = 41530.694
Completed tasks:                         mean = 743.200 	 std = 6.925
Total Task Time (incl. Reallocation):    mean = 8206.900 	 std = 10.024
Avg Task Time (incl. Reallocation):      mean = 11.044 	 std = 0.109
Std Task Time (incl. Reallocation):      mean = 3.391 	 std = 0.141
Total Service Time:                      mean = 4026.900 	 std = 34.125
Avg Service Time:                        mean = 5.419 	 std = 0.060
Std Service Time:                        mean = 2.469 	 std = 0.069
Avg Service Time Increase (%):           mean = 13.173% 	 std = 0.935%
=====================================================

=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:                                mean = 675210.300 	 std = 29136.579
Completed tasks:                         mean = 807.100 	 std = 7.203
Total Task Time (incl. Reallocation):    mean = 9022.100 	 std = 23.776
Avg Task Time (incl. Reallocation):      mean = 11.179 	 std = 0.108
Std Task Time (incl. Reallocation):      mean = 3.467 	 std = 0.161
Total Service Time:                      mean = 4478.700 	 std = 57.044
Avg Service Time:                        mean = 5.549 	 std = 0.071
Std Service Time:                        mean = 2.562 	 std = 0.107
Avg Service Time Increase (%):           mean = 15.749% 	 std = 1.101%
=====================================================

=====================================================
Experiment Results [1 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 3627.200 	 std = 95.272
Completed tasks:                         mean = 85.200 	 std = 2.600
Total Task Time (incl. Reallocation):    mean = 910.500 	 std = 4.455
Avg Task Time (incl. Reallocation):      mean = 10.696 	 std = 0.327
Std Task Time (incl. Reallocation):      mean = 2.852 	 std = 0.176
Total Service Time:                      mean = 403.300 	 std = 11.091
Avg Service Time:                        mean = 4.739 	 std = 0.209
Std Service Time:                        mean = 1.933 	 std = 0.105
Avg Service Time Increase (%):           mean = 0.000% 	 std = 0.000%
=====================================================

=====================================================
Experiment Results [2 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 12143.600 	 std = 910.654
Completed tasks:                         mean = 167.800 	 std = 2.441
Total Task Time (incl. Reallocation):    mean = 1820.200 	 std = 6.226
Avg Task Time (incl. Reallocation):      mean = 10.850 	 std = 0.185
Std Task Time (incl. Reallocation):      mean = 2.730 	 std = 0.141
Total Service Time:                      mean = 825.500 	 std = 15.628
Avg Service Time:                        mean = 4.921 	 std = 0.135
Std Service Time:                        mean = 1.948 	 std = 0.085
Avg Service Time Increase (%):           mean = 0.983% 	 std = 0.308%
=====================================================

=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 25941.500 	 std = 2191.787
Completed tasks:                         mean = 253.700 	 std = 4.337
Total Task Time (incl. Reallocation):    mean = 2734.300 	 std = 7.100
Avg Task Time (incl. Reallocation):      mean = 10.781 	 std = 0.206
Std Task Time (incl. Reallocation):      mean = 2.884 	 std = 0.128
Total Service Time:                      mean = 1270.000 	 std = 22.755
Avg Service Time:                        mean = 5.008 	 std = 0.155
Std Service Time:                        mean = 1.993 	 std = 0.089
Avg Service Time Increase (%):           mean = 2.521% 	 std = 0.644%
=====================================================

=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 52054.200 	 std = 10376.302
Completed tasks:                         mean = 341.600 	 std = 4.454
Total Task Time (incl. Reallocation):    mean = 3642.500 	 std = 4.801
Avg Task Time (incl. Reallocation):      mean = 10.665 	 std = 0.146
Std Task Time (incl. Reallocation):      mean = 2.919 	 std = 0.130
Total Service Time:                      mean = 1702.500 	 std = 14.361
Avg Service Time:                        mean = 4.985 	 std = 0.078
Std Service Time:                        mean = 2.017 	 std = 0.061
Avg Service Time Increase (%):           mean = 4.116% 	 std = 0.599%
=====================================================

=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 88674.500 	 std = 16153.721
Completed tasks:                         mean = 424.600 	 std = 3.878
Total Task Time (incl. Reallocation):    mean = 4546.100 	 std = 5.300
Avg Task Time (incl. Reallocation):      mean = 10.708 	 std = 0.104
Std Task Time (incl. Reallocation):      mean = 3.003 	 std = 0.108
Total Service Time:                      mean = 2142.900 	 std = 21.154
Avg Service Time:                        mean = 5.047 	 std = 0.069
Std Service Time:                        mean = 2.090 	 std = 0.068
Avg Service Time Increase (%):           mean = 5.227% 	 std = 1.062%
=====================================================

=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 148996.700 	 std = 16420.348
Completed tasks:                         mean = 504.600 	 std = 5.044
Total Task Time (incl. Reallocation):    mean = 5465.200 	 std = 10.058
Avg Task Time (incl. Reallocation):      mean = 10.832 	 std = 0.124
Std Task Time (incl. Reallocation):      mean = 3.067 	 std = 0.121
Total Service Time:                      mean = 2609.300 	 std = 35.134
Avg Service Time:                        mean = 5.171 	 std = 0.076
Std Service Time:                        mean = 2.179 	 std = 0.044
Avg Service Time Increase (%):           mean = 7.805% 	 std = 0.850%
=====================================================

=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 236957.200 	 std = 25945.964
Completed tasks:                         mean = 585.800 	 std = 6.145
Total Task Time (incl. Reallocation):    mean = 6380.400 	 std = 11.218
Avg Task Time (incl. Reallocation):      mean = 10.893 	 std = 0.128
Std Task Time (incl. Reallocation):      mean = 3.180 	 std = 0.102
Total Service Time:                      mean = 3071.800 	 std = 44.081
Avg Service Time:                        mean = 5.244 	 std = 0.093
Std Service Time:                        mean = 2.225 	 std = 0.062
Avg Service Time Increase (%):           mean = 9.522% 	 std = 1.129%
=====================================================

=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 395884.200 	 std = 58398.930
Completed tasks:                         mean = 657.800 	 std = 6.600
Total Task Time (incl. Reallocation):    mean = 7306.200 	 std = 12.424
Avg Task Time (incl. Reallocation):      mean = 11.108 	 std = 0.120
Std Task Time (incl. Reallocation):      mean = 3.338 	 std = 0.114
Total Service Time:                      mean = 3571.100 	 std = 24.753
Avg Service Time:                        mean = 5.429 	 std = 0.070
Std Service Time:                        mean = 2.346 	 std = 0.052
Avg Service Time Increase (%):           mean = 13.018% 	 std = 1.037%
=====================================================

=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 598178.800 	 std = 80569.273
Completed tasks:                         mean = 734.800 	 std = 9.867
Total Task Time (incl. Reallocation):    mean = 8218.400 	 std = 13.987
Avg Task Time (incl. Reallocation):      mean = 11.187 	 std = 0.158
Std Task Time (incl. Reallocation):      mean = 3.409 	 std = 0.113
Total Service Time:                      mean = 4038.000 	 std = 49.024
Avg Service Time:                        mean = 5.497 	 std = 0.115
Std Service Time:                        mean = 2.448 	 std = 0.051
Avg Service Time Increase (%):           mean = 14.952% 	 std = 1.425%
=====================================================

=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:                                mean = 882634.300 	 std = 77188.393
Completed tasks:                         mean = 794.300 	 std = 10.479
Total Task Time (incl. Reallocation):    mean = 9033.700 	 std = 20.110
Avg Task Time (incl. Reallocation):      mean = 11.375 	 std = 0.162
Std Task Time (incl. Reallocation):      mean = 3.597 	 std = 0.104
Total Service Time:                      mean = 4491.400 	 std = 55.446
Avg Service Time:                        mean = 5.655 	 std = 0.076
Std Service Time:                        mean = 2.652 	 std = 0.064
Avg Service Time Increase (%):           mean = 18.621% 	 std = 0.953%
=====================================================


A* Calls
| Algorithm                          | 1             | 2               | 3                | 4                 | 5                 | 6                  | 7                  | 8                  | 9                  | 10                   |
| ---------------------------------- | ------------- | --------------- | ---------------- | ----------------- | ----------------- | ------------------ | ------------------ | ------------------ | ------------------ | -------------------- |
| DECENTRALIZED_RESPECT              | 3627.2 (95.3) | 8616.4 (398.2)  | 14903.6 (515.1)  | 23603.7 (1281.8)  | 33813.4 (2082.8)  | 49391.7 (2447.0)   | 71365.0 (5322.2)   | 93859.8 (4993.3)   | 137436.5 (14004.0) | 192568.9 (17932.3)   |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 3627.2 (95.3) | 12704.4 (829.5) | 28355.7 (2865.0) | 56836.8 (11385.1) | 96837.1 (11593.6) | 174789.3 (32285.6) | 271263.2 (40784.5) | 487585.6 (49257.5) | 743421.2 (55166.6) | 1183481.5 (122521.2) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 3627.2 (95.3) | 11362.4 (614.4) | 25519.7 (1942.7) | 48669.7 (5586.4)  | 80443.6 (9141.0)  | 127094.9 (12860.3) | 206782.5 (17366.5) | 324765.9 (35047.1) | 491146.6 (41530.7) | 675210.3 (29136.6)   |
| DECENTRALIZED_NEGOTIATE_KARMA      | 3627.2 (95.3) | 12143.6 (910.7) | 25941.5 (2191.8) | 52054.2 (10376.3) | 88674.5 (16153.7) | 148996.7 (16420.3) | 236957.2 (25946.0) | 395884.2 (58398.9) | 598178.8 (80569.3) | 882634.3 (77188.4)   |

Completed Tasks
| Algorithm                          | 1          | 2           | 3           | 4           | 5           | 6           | 7           | 8           | 9           | 10           |
| ---------------------------------- | ---------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------------ |
| DECENTRALIZED_RESPECT              | 85.2 (2.6) | 166.5 (3.8) | 247.7 (3.0) | 325.4 (6.1) | 401.9 (4.2) | 472.9 (6.5) | 538.5 (7.8) | 604.1 (9.7) | 654.6 (9.0) | 701.7 (7.8)  |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 85.2 (2.6) | 168.7 (3.3) | 252.4 (3.6) | 340.3 (5.3) | 418.8 (5.6) | 499.3 (5.6) | 581.3 (5.2) | 656.5 (8.8) | 725.3 (8.4) | 783.8 (7.5)  |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 85.2 (2.6) | 170.8 (3.1) | 256.3 (5.4) | 342.1 (5.0) | 425.3 (4.1) | 510.0 (5.1) | 591.7 (8.1) | 669.3 (8.9) | 743.2 (6.9) | 807.1 (7.2)  |
| DECENTRALIZED_NEGOTIATE_KARMA      | 85.2 (2.6) | 167.8 (2.4) | 253.7 (4.3) | 341.6 (4.5) | 424.6 (3.9) | 504.6 (5.0) | 585.8 (6.1) | 657.8 (6.6) | 734.8 (9.9) | 794.3 (10.5) |

Total Task Time (incl. Reallocation)
| Algorithm                          | 1           | 2            | 3            | 4             | 5             | 6             | 7             | 8             | 9             | 10            |
| ---------------------------------- | ----------- | ------------ | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| DECENTRALIZED_RESPECT              | 910.5 (4.5) | 1822.0 (7.9) | 2737.0 (6.0) | 3657.4 (7.3)  | 4575.6 (6.1)  | 5496.5 (7.4)  | 6422.7 (16.8) | 7340.7 (11.2) | 8294.4 (18.7) | 9152.2 (23.6) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 910.5 (4.5) | 1822.2 (6.2) | 2732.3 (6.8) | 3641.2 (11.7) | 4559.5 (7.2)  | 5470.8 (11.7) | 6383.8 (10.3) | 7302.2 (11.7) | 8227.6 (14.8) | 9065.8 (10.6) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 910.5 (4.5) | 1820.2 (5.4) | 2726.4 (9.3) | 3639.9 (7.7)  | 4551.7 (10.9) | 5460.8 (9.7)  | 6375.7 (14.0) | 7286.8 (19.2) | 8206.9 (10.0) | 9022.1 (23.8) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 910.5 (4.5) | 1820.2 (6.2) | 2734.3 (7.1) | 3642.5 (4.8)  | 4546.1 (5.3)  | 5465.2 (10.1) | 6380.4 (11.2) | 7306.2 (12.4) | 8218.4 (14.0) | 9033.7 (20.1) |

Avg Task Time (incl. Reallocation)
| Algorithm                          | 1            | 2            | 3            | 4            | 5            | 6            | 7            | 8            | 9            | 10           |
| ---------------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| DECENTRALIZED_RESPECT              | 10.70 (0.33) | 10.95 (0.27) | 11.05 (0.15) | 11.24 (0.22) | 11.39 (0.12) | 11.63 (0.17) | 11.93 (0.19) | 12.15 (0.19) | 12.67 (0.19) | 13.04 (0.17) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 10.70 (0.33) | 10.81 (0.24) | 10.83 (0.16) | 10.70 (0.20) | 10.89 (0.15) | 10.96 (0.14) | 10.98 (0.10) | 11.13 (0.16) | 11.35 (0.15) | 11.57 (0.11) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 10.70 (0.33) | 10.66 (0.21) | 10.64 (0.25) | 10.64 (0.17) | 10.70 (0.12) | 10.71 (0.11) | 10.78 (0.15) | 10.89 (0.17) | 11.04 (0.11) | 11.18 (0.11) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 10.70 (0.33) | 10.85 (0.19) | 10.78 (0.21) | 10.66 (0.15) | 10.71 (0.10) | 10.83 (0.12) | 10.89 (0.13) | 11.11 (0.12) | 11.19 (0.16) | 11.38 (0.16) |

Std Task Time (incl. Reallocation)
| Algorithm                          | 1           | 2           | 3           | 4           | 5           | 6           | 7           | 8           | 9           | 10          |
| ---------------------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| DECENTRALIZED_RESPECT              | 2.85 (0.18) | 2.81 (0.12) | 3.05 (0.08) | 3.21 (0.11) | 3.37 (0.08) | 3.48 (0.12) | 3.73 (0.11) | 3.94 (0.11) | 4.26 (0.10) | 4.46 (0.10) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 2.85 (0.18) | 2.69 (0.15) | 2.92 (0.12) | 2.93 (0.10) | 3.04 (0.09) | 3.10 (0.12) | 3.18 (0.09) | 3.27 (0.10) | 3.49 (0.06) | 3.62 (0.12) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 2.85 (0.18) | 2.82 (0.19) | 2.84 (0.10) | 2.90 (0.13) | 2.97 (0.10) | 3.04 (0.04) | 3.12 (0.11) | 3.30 (0.12) | 3.39 (0.14) | 3.47 (0.16) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 2.85 (0.18) | 2.73 (0.14) | 2.88 (0.13) | 2.92 (0.13) | 3.00 (0.11) | 3.07 (0.12) | 3.18 (0.10) | 3.34 (0.11) | 3.41 (0.11) | 3.60 (0.10) |

Total Service Time
| Algorithm                          | 1            | 2            | 3             | 4             | 5             | 6             | 7             | 8             | 9             | 10            |
| ---------------------------------- | ------------ | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| DECENTRALIZED_RESPECT              | 403.3 (11.1) | 835.2 (21.7) | 1288.3 (22.8) | 1742.1 (22.8) | 2223.3 (26.2) | 2726.1 (31.2) | 3260.8 (32.9) | 3762.7 (54.3) | 4340.7 (35.3) | 4919.4 (28.5) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 403.3 (11.1) | 836.0 (21.7) | 1262.9 (14.1) | 1700.5 (21.1) | 2147.6 (24.7) | 2596.9 (15.9) | 3070.3 (23.8) | 3540.4 (27.4) | 4047.7 (37.1) | 4450.6 (51.2) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 403.3 (11.1) | 820.2 (14.6) | 1253.9 (19.6) | 1695.1 (34.5) | 2164.3 (25.1) | 2629.0 (31.9) | 3074.7 (33.8) | 3563.7 (52.6) | 4026.9 (34.1) | 4478.7 (57.0) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 403.3 (11.1) | 825.5 (15.6) | 1270.0 (22.8) | 1702.5 (14.4) | 2142.9 (21.2) | 2609.3 (35.1) | 3071.8 (44.1) | 3571.1 (24.8) | 4038.0 (49.0) | 4491.4 (55.4) |

Avg Service Time
| Algorithm                          | 1           | 2           | 3           | 4           | 5           | 6           | 7           | 8           | 9           | 10          |
| ---------------------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| DECENTRALIZED_RESPECT              | 4.74 (0.21) | 5.02 (0.21) | 5.20 (0.08) | 5.36 (0.11) | 5.53 (0.07) | 5.77 (0.11) | 6.06 (0.12) | 6.23 (0.12) | 6.63 (0.11) | 7.01 (0.08) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 4.74 (0.21) | 4.96 (0.19) | 5.00 (0.09) | 5.00 (0.10) | 5.13 (0.10) | 5.20 (0.06) | 5.28 (0.08) | 5.39 (0.11) | 5.58 (0.09) | 5.68 (0.08) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 4.74 (0.21) | 4.80 (0.14) | 4.89 (0.12) | 4.96 (0.12) | 5.09 (0.05) | 5.16 (0.10) | 5.20 (0.10) | 5.33 (0.13) | 5.42 (0.06) | 5.55 (0.07) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 4.74 (0.21) | 4.92 (0.13) | 5.01 (0.15) | 4.98 (0.08) | 5.05 (0.07) | 5.17 (0.08) | 5.24 (0.09) | 5.43 (0.07) | 5.50 (0.12) | 5.66 (0.08) |

Std Service Time
| Algorithm                          | 1           | 2           | 3           | 4           | 5           | 6           | 7           | 8           | 9           | 10          |
| ---------------------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| DECENTRALIZED_RESPECT              | 1.93 (0.10) | 2.02 (0.12) | 2.10 (0.11) | 2.26 (0.08) | 2.38 (0.06) | 2.55 (0.07) | 2.79 (0.10) | 2.97 (0.08) | 3.20 (0.09) | 3.46 (0.06) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 1.93 (0.10) | 1.93 (0.07) | 2.00 (0.10) | 2.04 (0.10) | 2.07 (0.04) | 2.15 (0.05) | 2.27 (0.08) | 2.33 (0.07) | 2.47 (0.04) | 2.59 (0.07) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 1.93 (0.10) | 1.94 (0.15) | 1.97 (0.07) | 2.01 (0.08) | 2.14 (0.06) | 2.17 (0.05) | 2.27 (0.06) | 2.35 (0.08) | 2.47 (0.07) | 2.56 (0.11) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 1.93 (0.10) | 1.95 (0.08) | 1.99 (0.09) | 2.02 (0.06) | 2.09 (0.07) | 2.18 (0.04) | 2.22 (0.06) | 2.35 (0.05) | 2.45 (0.05) | 2.65 (0.06) |

Avg Service Time Increase (%)
| Algorithm                          | 1           | 2           | 3           | 4            | 5            | 6            | 7            | 8            | 9            | 10           |
| ---------------------------------- | ----------- | ----------- | ----------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| DECENTRALIZED_RESPECT              | 0.00 (0.00) | 3.49 (1.20) | 7.46 (1.52) | 11.76 (0.83) | 15.36 (0.67) | 21.12 (1.19) | 26.55 (1.66) | 31.51 (1.96) | 37.80 (1.22) | 44.60 (1.42) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 0.00 (0.00) | 1.41 (0.72) | 3.21 (0.64) | 4.83 (0.64)  | 6.98 (0.86)  | 9.41 (1.45)  | 12.05 (0.81) | 14.69 (1.45) | 17.66 (1.23) | 21.02 (1.15) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 0.00 (0.00) | 0.86 (0.26) | 2.10 (0.47) | 3.43 (0.63)  | 5.17 (0.47)  | 6.63 (0.80)  | 8.96 (0.76)  | 11.02 (0.97) | 13.17 (0.93) | 15.75 (1.10) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 0.00 (0.00) | 0.98 (0.31) | 2.52 (0.64) | 4.12 (0.60)  | 5.23 (1.06)  | 7.80 (0.85)  | 9.52 (1.13)  | 13.02 (1.04) | 14.95 (1.42) | 18.62 (0.95) |
"""
