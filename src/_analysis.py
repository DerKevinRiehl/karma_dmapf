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

random_seeds = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

grid_sizes = [5]  # , 10, 15, 20]

n_agents_map = {
    5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    15: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    20: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
for grid_size in grid_sizes:
    for controller in [
        # MAPF_CONTROLLER_CENTRALIZED,
        MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
        MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA,
    ]:

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
            n_total_cost_list = []
            n_average_cost_list = []
            n_distribution_list = []

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

                # evaluation
                task_completion_times = [
                    task.completed_time - task.spawned_time
                    for task in environment.completed_tasks
                    if task.completed_time is not None
                ]
                n_completed_tasks = len(task_completion_times)
                n_total_cost = np.sum(task_completion_times)
                n_average_cost = np.mean(task_completion_times)
                n_distribution = np.std(task_completion_times)
                # store metrics
                astar_calls_list.append(n_astar_calls)
                n_completed_tasks_list.append(n_completed_tasks)
                n_total_cost_list.append(n_total_cost)
                n_average_cost_list.append(n_average_cost)
                n_distribution_list.append(n_distribution)

            def summarize(x):
                x = np.array(x, dtype=float)
                return x.mean(), x.std(
                    ddof=0
                )  # population std; use ddof=1 for sample std

            avg_astar, std_astar = summarize(astar_calls_list)
            avg_completed, std_completed = summarize(n_completed_tasks_list)
            avg_total_cost, std_total_cost = summarize(n_total_cost_list)
            avg_avg_cost, std_avg_cost = summarize(n_average_cost_list)
            avg_distribution, std_distribution = summarize(n_distribution_list)

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
                "A* calls:        mean = {:.3f} \t std = {:.3f}".format(
                    avg_astar, std_astar
                )
            )
            print(
                "Completed tasks: mean = {:.3f} \t std = {:.3f}".format(
                    avg_completed, std_completed
                )
            )
            print(
                "Total cost:      mean = {:.3f} \t std = {:.3f}".format(
                    avg_total_cost, std_total_cost
                )
            )
            print(
                "Avg cost:        mean = {:.3f} \t std = {:.3f}".format(
                    avg_avg_cost, std_avg_cost
                )
            )
            print(
                "Distribution:    mean = {:.3f} \t std = {:.3f}".format(
                    avg_distribution, std_distribution
                )
            )
            print("=====================================================")

"""
=====================================================
Experiment Results [1 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 3627.200 	 std = 95.272
Completed tasks: mean = 85.200 	 std = 2.600
Total cost:      mean = 910.500 	 std = 4.455
Avg cost:        mean = 10.696 	 std = 0.327
Distribution:    mean = 2.852 	 std = 0.176
=====================================================
=====================================================
Experiment Results [2 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 8616.400 	 std = 398.214
Completed tasks: mean = 166.500 	 std = 3.775
Total cost:      mean = 1822.000 	 std = 7.887
Avg cost:        mean = 10.949 	 std = 0.266
Distribution:    mean = 2.814 	 std = 0.122
=====================================================
=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 14903.600 	 std = 515.064
Completed tasks: mean = 247.700 	 std = 2.968
Total cost:      mean = 2737.000 	 std = 6.033
Avg cost:        mean = 11.051 	 std = 0.151
Distribution:    mean = 3.048 	 std = 0.078
=====================================================
=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 23603.700 	 std = 1281.768
Completed tasks: mean = 325.400 	 std = 6.070
Total cost:      mean = 3657.400 	 std = 7.338
Avg cost:        mean = 11.244 	 std = 0.223
Distribution:    mean = 3.214 	 std = 0.113
=====================================================
=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 33813.400 	 std = 2082.818
Completed tasks: mean = 401.900 	 std = 4.182
Total cost:      mean = 4575.600 	 std = 6.070
Avg cost:        mean = 11.386 	 std = 0.120
Distribution:    mean = 3.370 	 std = 0.080
=====================================================
=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 49391.700 	 std = 2446.961
Completed tasks: mean = 472.900 	 std = 6.534
Total cost:      mean = 5496.500 	 std = 7.420
Avg cost:        mean = 11.625 	 std = 0.167
Distribution:    mean = 3.477 	 std = 0.118
=====================================================
=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 71365.000 	 std = 5322.212
Completed tasks: mean = 538.500 	 std = 7.826
Total cost:      mean = 6422.700 	 std = 16.769
Avg cost:        mean = 11.930 	 std = 0.187
Distribution:    mean = 3.729 	 std = 0.114
=====================================================
=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 93859.800 	 std = 4993.255
Completed tasks: mean = 604.100 	 std = 9.669
Total cost:      mean = 7340.700 	 std = 11.172
Avg cost:        mean = 12.155 	 std = 0.192
Distribution:    mean = 3.942 	 std = 0.113
=====================================================
=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 137436.500 	 std = 14003.956
Completed tasks: mean = 654.600 	 std = 8.969
Total cost:      mean = 8294.400 	 std = 18.688
Avg cost:        mean = 12.673 	 std = 0.185
Distribution:    mean = 4.264 	 std = 0.097
=====================================================
=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 192568.900 	 std = 17932.290
Completed tasks: mean = 701.700 	 std = 7.849
Total cost:      mean = 9152.200 	 std = 23.613
Avg cost:        mean = 13.045 	 std = 0.170
Distribution:    mean = 4.456 	 std = 0.099
=====================================================
"""


"""
=====================================================
Experiment Results [1 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 3627.200 	 std = 95.272
Completed tasks: mean = 85.200 	 std = 2.600
Total cost:      mean = 910.500 	 std = 4.455
Avg cost:        mean = 10.696 	 std = 0.327
Distribution:    mean = 2.852 	 std = 0.176
=====================================================
=====================================================
Experiment Results [2 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 12704.400 	 std = 829.455
Completed tasks: mean = 168.700 	 std = 3.318
Total cost:      mean = 1822.200 	 std = 6.210
Avg cost:        mean = 10.806 	 std = 0.245
Distribution:    mean = 2.692 	 std = 0.151
=====================================================
=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 28355.700 	 std = 2865.048
Completed tasks: mean = 252.400 	 std = 3.555
Total cost:      mean = 2732.300 	 std = 6.827
Avg cost:        mean = 10.828 	 std = 0.163
Distribution:    mean = 2.922 	 std = 0.115
=====================================================
=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 56836.800 	 std = 11385.130
Completed tasks: mean = 340.300 	 std = 5.292
Total cost:      mean = 3641.200 	 std = 11.677
Avg cost:        mean = 10.703 	 std = 0.200
Distribution:    mean = 2.928 	 std = 0.104
=====================================================
=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 96837.100 	 std = 11593.635
Completed tasks: mean = 418.800 	 std = 5.582
Total cost:      mean = 4559.500 	 std = 7.242
Avg cost:        mean = 10.889 	 std = 0.154
Distribution:    mean = 3.043 	 std = 0.088
=====================================================
=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 174789.300 	 std = 32285.563
Completed tasks: mean = 499.300 	 std = 5.640
Total cost:      mean = 5470.800 	 std = 11.746
Avg cost:        mean = 10.958 	 std = 0.137
Distribution:    mean = 3.098 	 std = 0.116
=====================================================
=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 271263.200 	 std = 40784.480
Completed tasks: mean = 581.300 	 std = 5.178
Total cost:      mean = 6383.800 	 std = 10.255
Avg cost:        mean = 10.983 	 std = 0.104
Distribution:    mean = 3.175 	 std = 0.086
=====================================================
=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 487585.600 	 std = 49257.502
Completed tasks: mean = 656.500 	 std = 8.801
Total cost:      mean = 7302.200 	 std = 11.712
Avg cost:        mean = 11.125 	 std = 0.159
Distribution:    mean = 3.274 	 std = 0.100
=====================================================
=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 743421.200 	 std = 55166.601
Completed tasks: mean = 725.300 	 std = 8.427
Total cost:      mean = 8227.600 	 std = 14.800
Avg cost:        mean = 11.345 	 std = 0.147
Distribution:    mean = 3.489 	 std = 0.063
=====================================================
=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 1183481.500 	 std = 122521.154
Completed tasks: mean = 783.800 	 std = 7.494
Total cost:      mean = 9065.800 	 std = 10.600
Avg cost:        mean = 11.568 	 std = 0.112
Distribution:    mean = 3.625 	 std = 0.123
=====================================================
"""


"""
=====================================================
Experiment Results [1 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 3627.200 	 std = 95.272
Completed tasks: mean = 85.200 	 std = 2.600
Total cost:      mean = 910.500 	 std = 4.455
Avg cost:        mean = 10.696 	 std = 0.327
Distribution:    mean = 2.852 	 std = 0.176
=====================================================
=====================================================
Experiment Results [2 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 11362.400 	 std = 614.417
Completed tasks: mean = 170.800 	 std = 3.092
Total cost:      mean = 1820.200 	 std = 5.381
Avg cost:        mean = 10.661 	 std = 0.211
Distribution:    mean = 2.816 	 std = 0.189
=====================================================
=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 25519.700 	 std = 1942.669
Completed tasks: mean = 256.300 	 std = 5.367
Total cost:      mean = 2726.400 	 std = 9.276
Avg cost:        mean = 10.643 	 std = 0.249
Distribution:    mean = 2.843 	 std = 0.098
=====================================================
=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 48669.700 	 std = 5586.435
Completed tasks: mean = 342.100 	 std = 5.009
Total cost:      mean = 3639.900 	 std = 7.687
Avg cost:        mean = 10.642 	 std = 0.171
Distribution:    mean = 2.898 	 std = 0.127
=====================================================
=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 80443.600 	 std = 9141.000
Completed tasks: mean = 425.300 	 std = 4.124
Total cost:      mean = 4551.700 	 std = 10.946
Avg cost:        mean = 10.703 	 std = 0.118
Distribution:    mean = 2.970 	 std = 0.095
=====================================================
=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 127094.900 	 std = 12860.332
Completed tasks: mean = 510.000 	 std = 5.138
Total cost:      mean = 5460.800 	 std = 9.673
Avg cost:        mean = 10.709 	 std = 0.113
Distribution:    mean = 3.037 	 std = 0.042
=====================================================
=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 206782.500 	 std = 17366.528
Completed tasks: mean = 591.700 	 std = 8.125
Total cost:      mean = 6375.700 	 std = 13.957
Avg cost:        mean = 10.777 	 std = 0.152
Distribution:    mean = 3.118 	 std = 0.111
=====================================================
=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 324765.900 	 std = 35047.142
Completed tasks: mean = 669.300 	 std = 8.855
Total cost:      mean = 7286.800 	 std = 19.167
Avg cost:        mean = 10.889 	 std = 0.166
Distribution:    mean = 3.297 	 std = 0.117
=====================================================
=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 491146.600 	 std = 41530.694
Completed tasks: mean = 743.200 	 std = 6.925
Total cost:      mean = 8206.900 	 std = 10.024
Avg cost:        mean = 11.044 	 std = 0.109
Distribution:    mean = 3.391 	 std = 0.141
=====================================================
=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 675210.300 	 std = 29136.579
Completed tasks: mean = 807.100 	 std = 7.203
Total cost:      mean = 9022.100 	 std = 23.776
Avg cost:        mean = 11.179 	 std = 0.108
Distribution:    mean = 3.467 	 std = 0.161
=====================================================
"""


"""
=====================================================
Experiment Results [1 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 3627.200 	 std = 95.272
Completed tasks: mean = 85.200 	 std = 2.600
Total cost:      mean = 910.500 	 std = 4.455
Avg cost:        mean = 10.696 	 std = 0.327
Distribution:    mean = 2.852 	 std = 0.176
=====================================================
=====================================================
Experiment Results [2 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 12143.600 	 std = 910.654
Completed tasks: mean = 167.800 	 std = 2.441
Total cost:      mean = 1820.200 	 std = 6.226
Avg cost:        mean = 10.850 	 std = 0.185
Distribution:    mean = 2.730 	 std = 0.141
=====================================================
=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 25941.500 	 std = 2191.787
Completed tasks: mean = 253.700 	 std = 4.337
Total cost:      mean = 2734.300 	 std = 7.100
Avg cost:        mean = 10.781 	 std = 0.206
Distribution:    mean = 2.884 	 std = 0.128
=====================================================
=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 52054.200 	 std = 10376.302
Completed tasks: mean = 341.600 	 std = 4.454
Total cost:      mean = 3642.500 	 std = 4.801
Avg cost:        mean = 10.665 	 std = 0.146
Distribution:    mean = 2.919 	 std = 0.130
=====================================================
=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 88674.500 	 std = 16153.721
Completed tasks: mean = 424.600 	 std = 3.878
Total cost:      mean = 4546.100 	 std = 5.300
Avg cost:        mean = 10.708 	 std = 0.104
Distribution:    mean = 3.003 	 std = 0.108
=====================================================
=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 148996.700 	 std = 16420.348
Completed tasks: mean = 504.600 	 std = 5.044
Total cost:      mean = 5465.200 	 std = 10.058
Avg cost:        mean = 10.832 	 std = 0.124
Distribution:    mean = 3.067 	 std = 0.121
=====================================================
=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 236957.200 	 std = 25945.964
Completed tasks: mean = 585.800 	 std = 6.145
Total cost:      mean = 6380.400 	 std = 11.218
Avg cost:        mean = 10.893 	 std = 0.128
Distribution:    mean = 3.180 	 std = 0.102
=====================================================
=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 395884.200 	 std = 58398.930
Completed tasks: mean = 657.800 	 std = 6.600
Total cost:      mean = 7306.200 	 std = 12.424
Avg cost:        mean = 11.108 	 std = 0.120
Distribution:    mean = 3.338 	 std = 0.114
=====================================================
=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 598178.800 	 std = 80569.273
Completed tasks: mean = 734.800 	 std = 9.867
Total cost:      mean = 8218.400 	 std = 13.987
Avg cost:        mean = 11.187 	 std = 0.158
Distribution:    mean = 3.409 	 std = 0.113
=====================================================
=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 882634.300 	 std = 77188.393
Completed tasks: mean = 794.300 	 std = 10.479
Total cost:      mean = 9033.700 	 std = 20.110
Avg cost:        mean = 11.375 	 std = 0.162
Distribution:    mean = 3.597 	 std = 0.104
=====================================================
"""

# ! TABLE BELOW IS OUTDATED - code for generation is missing from this script; refer to the printed results above for the most up-to-date metrics
"""
A* Calls
| Algorithm                          | 1             | 2               | 3                | 4                 | 5                 | 6                  | 7                  | 8                  | 9                  | 10                   |
| ---------------------------------- | ------------- | --------------- | ---------------- | ----------------- | ----------------- | ------------------ | ------------------ | ------------------ | ------------------ | -------------------- |
| DECENTRALIZED_RESPECT              | 3627.2 (95.3) | 8616.4 (398.2)  | 14903.6 (515.1)  | 23603.7 (1281.8)  | 33813.4 (2082.8)  | 49391.7 (2447.0)   | 71365.0 (5322.2)   | 93859.8 (4993.3)   | 137436.5 (14004.0) | 192568.9 (17932.3)   |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 3627.2 (95.3) | 11827.1 (588.4) | 24001.0 (2077.5) | 57626.1 (19747.2) | 85014.5 (16216.9) | 159407.4 (36010.8) | 282301.2 (42488.5) | 460835.5 (74604.5) | 672043.0 (88455.7) | 1092802.2 (102441.0) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 3627.2 (95.3) | 10971.6 (554.2) | 22421.4 (1728.5) | 41284.0 (2358.1)  | 71787.9 (9656.9)  | 117762.1 (11983.3) | 190382.8 (14146.9) | 286685.2 (22163.8) | 428890.9 (36802.2) | 643188.6 (52284.9)   |
| DECENTRALIZED_NEGOTIATE_KARMA      | 3627.2 (95.3) | 11407.3 (809.6) | 23376.1 (1565.1) | 43177.0 (4783.2)  | 77670.0 (17932.1) | 132168.7 (26922.5) | 230870.5 (37196.6) | 377555.7 (43255.6) | 526167.7 (77633.3) | 816255.7 (95092.4)   |

Completed Tasks
| Algorithm                          | 1          | 2           | 3           | 4           | 5           | 6           | 7           | 8           | 9            | 10           |
| ---------------------------------- | ---------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------------ | ------------ |
| DECENTRALIZED_RESPECT              | 85.2 (2.6) | 166.5 (3.8) | 247.7 (3.0) | 325.4 (6.1) | 401.9 (4.2) | 472.9 (6.5) | 538.5 (7.8) | 604.1 (9.7) | 654.6 (9.0)  | 701.7 (7.8)  |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 85.2 (2.6) | 168.0 (3.4) | 252.5 (2.8) | 340.1 (5.5) | 421.9 (4.6) | 505.4 (6.4) | 586.1 (6.7) | 657.8 (8.6) | 734.3 (5.4)  | 792.2 (7.4)  |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 85.2 (2.6) | 170.8 (3.2) | 256.9 (3.5) | 340.8 (6.3) | 426.9 (6.4) | 513.5 (4.7) | 595.0 (9.3) | 674.9 (6.1) | 752.4 (10.0) | 815.5 (11.7) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 85.2 (2.6) | 168.4 (2.5) | 256.3 (2.1) | 340.0 (4.3) | 425.8 (7.2) | 510.6 (8.0) | 589.4 (6.5) | 665.0 (7.8) | 746.8 (5.9)  | 803.8 (6.2)  |

Total Cost
| Algorithm                          | 1           | 2            | 3            | 4             | 5            | 6             | 7             | 8             | 9             | 10            |
| ---------------------------------- | ----------- | ------------ | ------------ | ------------- | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- |
| DECENTRALIZED_RESPECT              | 910.5 (4.5) | 1822.0 (7.9) | 2737.0 (6.0) | 3657.4 (7.3)  | 4575.6 (6.1) | 5496.5 (7.4)  | 6422.7 (16.8) | 7340.7 (11.2) | 8294.4 (18.7) | 9152.2 (23.6) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 910.5 (4.5) | 1822.2 (5.8) | 2734.6 (8.8) | 3646.4 (5.2)  | 4557.7 (5.8) | 5465.2 (14.6) | 6375.7 (10.9) | 7299.4 (13.5) | 8218.0 (16.6) | 9049.0 (26.1) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 910.5 (4.5) | 1818.9 (4.3) | 2732.0 (8.0) | 3636.1 (7.7)  | 4554.1 (7.6) | 5460.3 (7.5)  | 6371.5 (11.5) | 7284.6 (9.2)  | 8197.1 (11.0) | 9012.9 (25.2) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 910.5 (4.5) | 1821.8 (6.2) | 2726.1 (5.2) | 3638.3 (6.6)  | 4551.1 (7.2) | 5457.9 (7.0)  | 6375.0 (15.3) | 7291.8 (16.9) | 8203.3 (13.1) | 9023.2 (26.6) |

Avg Cost
| Algorithm                          | 1            | 2            | 3            | 4            | 5            | 6            | 7            | 8            | 9            | 10           |
| ---------------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| DECENTRALIZED_RESPECT              | 10.70 (0.33) | 10.95 (0.27) | 11.05 (0.15) | 11.24 (0.22) | 11.39 (0.12) | 11.63 (0.17) | 11.93 (0.19) | 12.16 (0.19) | 12.67 (0.19) | 13.05 (0.17) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 10.70 (0.33) | 10.85 (0.24) | 10.83 (0.15) | 10.73 (0.19) | 10.80 (0.12) | 10.82 (0.16) | 10.88 (0.13) | 11.10 (0.16) | 11.19 (0.08) | 11.42 (0.12) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 10.70 (0.33) | 10.65 (0.22) | 10.64 (0.16) | 10.67 (0.20) | 10.67 (0.17) | 10.63 (0.10) | 10.71 (0.18) | 10.79 (0.10) | 10.90 (0.15) | 11.06 (0.18) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 10.70 (0.33) | 10.82 (0.17) | 10.64 (0.08) | 10.70 (0.14) | 10.69 (0.19) | 10.69 (0.18) | 10.82 (0.14) | 10.97 (0.15) | 10.99 (0.10) | 11.23 (0.10) |

Distribution
| Algorithm                          | 1           | 2           | 3           | 4           | 5           | 6           | 7           | 8           | 9           | 10          |
| ---------------------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| DECENTRALIZED_RESPECT              | 2.85 (0.18) | 2.81 (0.12) | 3.05 (0.08) | 3.21 (0.11) | 3.37 (0.08) | 3.48 (0.12) | 3.73 (0.11) | 3.94 (0.11) | 4.26 (0.10) | 4.46 (0.10) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 2.85 (0.18) | 2.75 (0.10) | 2.86 (0.13) | 2.98 (0.13) | 2.97 (0.11) | 3.09 (0.08) | 3.19 (0.10) | 3.27 (0.06) | 3.45 (0.11) | 3.60 (0.09) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 2.85 (0.18) | 2.80 (0.18) | 2.91 (0.12) | 2.92 (0.11) | 3.04 (0.09) | 3.01 (0.10) | 3.19 (0.08) | 3.21 (0.13) | 3.29 (0.11) | 3.47 (0.17) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 2.85 (0.18) | 2.76 (0.09) | 2.93 (0.13) | 2.95 (0.07) | 2.93 (0.12) | 3.03 (0.10) | 3.18 (0.13) | 3.21 (0.10) | 3.34 (0.07) | 3.48 (0.11) |
"""
