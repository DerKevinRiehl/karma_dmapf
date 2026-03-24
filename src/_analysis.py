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

grid_sizes = [5]#, 10, 15, 20]

n_agents_map = {
    5: [1, 2, 3, 4, 5, ],#6, 7, 8, 9, 10],
    10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    15: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    20: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
for grid_size in grid_sizes:
    for controller in [
            # MAPF_CONTROLLER_CENTRALIZED,
            # MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
            # MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
            # MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
            MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA
        ]:
        
        for n_agent in n_agents_map[grid_size]:
            simulation_settings = {
                "random_seed": 42,
                "grid_size": grid_size+2,
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
                    "initial_karma": 5,
                    "lambda": 0.1,
                    "gamma": 0.1
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
                while environment.time < environment.settings["time_simulation_duration"]:            
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
                        if old_len == len(environment.tasks): # currently too crowded to spawn
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
                return x.mean(), x.std(ddof=0)  # population std; use ddof=1 for sample std
            
            avg_astar, std_astar = summarize(astar_calls_list)
            avg_completed, std_completed = summarize(n_completed_tasks_list)
            avg_total_cost, std_total_cost = summarize(n_total_cost_list)
            avg_avg_cost, std_avg_cost = summarize(n_average_cost_list)
            avg_distribution, std_distribution = summarize(n_distribution_list)
            
            print("=====================================================")
            print(
                "Experiment Results ["+str(simulation_settings["n_agents"])+" agents] ["+str(simulation_settings["grid_size"]-2)+" grid-size] for algorithm",
                simulation_settings["mapf_control"],
                "over",
                len(astar_calls_list),
                "experiments",
            )
            print("=====================================================")
            print("A* calls:        mean = {:.3f} \t std = {:.3f}".format(avg_astar, std_astar))
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
                "Avg cost:        mean = {:.3f} \t std = {:.3f}".format(avg_avg_cost, std_avg_cost)
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
A* calls:        mean = 11827.100 	 std = 588.394
Completed tasks: mean = 168.000 	 std = 3.435
Total cost:      mean = 1822.200 	 std = 5.758
Avg cost:        mean = 10.851 	 std = 0.238
Distribution:    mean = 2.751 	 std = 0.099
=====================================================

=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 24001.000 	 std = 2077.534
Completed tasks: mean = 252.500 	 std = 2.837
Total cost:      mean = 2734.600 	 std = 8.845
Avg cost:        mean = 10.832 	 std = 0.149
Distribution:    mean = 2.861 	 std = 0.134
=====================================================

=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 57626.100 	 std = 19747.181
Completed tasks: mean = 340.100 	 std = 5.522
Total cost:      mean = 3646.400 	 std = 5.238
Avg cost:        mean = 10.725 	 std = 0.186
Distribution:    mean = 2.979 	 std = 0.132
=====================================================

=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 85014.500 	 std = 16216.915
Completed tasks: mean = 421.900 	 std = 4.571
Total cost:      mean = 4557.700 	 std = 5.763
Avg cost:        mean = 10.804 	 std = 0.120
Distribution:    mean = 2.965 	 std = 0.112
=====================================================

=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 159407.400 	 std = 36010.773
Completed tasks: mean = 505.400 	 std = 6.406
Total cost:      mean = 5465.200 	 std = 14.600
Avg cost:        mean = 10.816 	 std = 0.159
Distribution:    mean = 3.094 	 std = 0.078
=====================================================

=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 282301.200 	 std = 42488.508
Completed tasks: mean = 586.100 	 std = 6.655
Total cost:      mean = 6375.700 	 std = 10.927
Avg cost:        mean = 10.880 	 std = 0.132
Distribution:    mean = 3.188 	 std = 0.102
=====================================================

=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 460835.500 	 std = 74604.517
Completed tasks: mean = 657.800 	 std = 8.600
Total cost:      mean = 7299.400 	 std = 13.537
Avg cost:        mean = 11.099 	 std = 0.157
Distribution:    mean = 3.273 	 std = 0.058
=====================================================

=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 672043.000 	 std = 88455.674
Completed tasks: mean = 734.300 	 std = 5.423
Total cost:      mean = 8218.000 	 std = 16.625
Avg cost:        mean = 11.192 	 std = 0.083
Distribution:    mean = 3.450 	 std = 0.113
=====================================================

=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 1092802.200 	 std = 102440.972
Completed tasks: mean = 792.200 	 std = 7.400
Total cost:      mean = 9049.000 	 std = 26.054
Avg cost:        mean = 11.424 	 std = 0.121
Distribution:    mean = 3.599 	 std = 0.090
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
A* calls:        mean = 10971.600 	 std = 554.168
Completed tasks: mean = 170.800 	 std = 3.219
Total cost:      mean = 1818.900 	 std = 4.253
Avg cost:        mean = 10.653 	 std = 0.217
Distribution:    mean = 2.799 	 std = 0.182
=====================================================

=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 22421.400 	 std = 1728.473
Completed tasks: mean = 256.900 	 std = 3.534
Total cost:      mean = 2732.000 	 std = 7.987
Avg cost:        mean = 10.637 	 std = 0.160
Distribution:    mean = 2.908 	 std = 0.121
=====================================================

=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 41284.000 	 std = 2358.090
Completed tasks: mean = 340.800 	 std = 6.274
Total cost:      mean = 3636.100 	 std = 7.674
Avg cost:        mean = 10.673 	 std = 0.203
Distribution:    mean = 2.922 	 std = 0.105
=====================================================

=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 71787.900 	 std = 9656.915
Completed tasks: mean = 426.900 	 std = 6.363
Total cost:      mean = 4554.100 	 std = 7.595
Avg cost:        mean = 10.670 	 std = 0.165
Distribution:    mean = 3.043 	 std = 0.086
=====================================================

=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 117762.100 	 std = 11983.270
Completed tasks: mean = 513.500 	 std = 4.696
Total cost:      mean = 5460.300 	 std = 7.524
Avg cost:        mean = 10.634 	 std = 0.103
Distribution:    mean = 3.006 	 std = 0.097
=====================================================

=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 190382.800 	 std = 14146.909
Completed tasks: mean = 595.000 	 std = 9.263
Total cost:      mean = 6371.500 	 std = 11.491
Avg cost:        mean = 10.711 	 std = 0.175
Distribution:    mean = 3.194 	 std = 0.083
=====================================================

=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 286685.200 	 std = 22163.828
Completed tasks: mean = 674.900 	 std = 6.057
Total cost:      mean = 7284.600 	 std = 9.178
Avg cost:        mean = 10.794 	 std = 0.100
Distribution:    mean = 3.205 	 std = 0.133
=====================================================

=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 428890.900 	 std = 36802.247
Completed tasks: mean = 752.400 	 std = 10.002
Total cost:      mean = 8197.100 	 std = 11.049
Avg cost:        mean = 10.897 	 std = 0.150
Distribution:    mean = 3.288 	 std = 0.108
=====================================================

=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 643188.600 	 std = 52284.917
Completed tasks: mean = 815.500 	 std = 11.741
Total cost:      mean = 9012.900 	 std = 25.173
Avg cost:        mean = 11.055 	 std = 0.182
Distribution:    mean = 3.465 	 std = 0.168
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
A* calls:        mean = 10742.300 	 std = 837.776
Completed tasks: mean = 168.800 	 std = 3.919
Total cost:      mean = 1821.600 	 std = 8.405
Avg cost:        mean = 10.798 	 std = 0.281
Distribution:    mean = 2.772 	 std = 0.130
=====================================================

=====================================================
Experiment Results [3 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 24109.400 	 std = 1534.891
Completed tasks: mean = 253.300 	 std = 3.579
Total cost:      mean = 2733.500 	 std = 5.714
Avg cost:        mean = 10.794 	 std = 0.154
Distribution:    mean = 2.957 	 std = 0.160
=====================================================

=====================================================
Experiment Results [4 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 44633.400 	 std = 4453.413
Completed tasks: mean = 336.400 	 std = 4.271
Total cost:      mean = 3644.900 	 std = 10.559
Avg cost:        mean = 10.837 	 std = 0.151
Distribution:    mean = 3.071 	 std = 0.117
=====================================================

=====================================================
Experiment Results [5 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 74146.100 	 std = 6520.695
Completed tasks: mean = 417.000 	 std = 4.775
Total cost:      mean = 4560.900 	 std = 6.252
Avg cost:        mean = 10.939 	 std = 0.126
Distribution:    mean = 3.163 	 std = 0.136
=====================================================

=====================================================
Experiment Results [6 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 128263.700 	 std = 7622.095
Completed tasks: mean = 500.900 	 std = 5.356
Total cost:      mean = 5468.000 	 std = 11.402
Avg cost:        mean = 10.918 	 std = 0.126
Distribution:    mean = 3.307 	 std = 0.128
=====================================================

=====================================================
Experiment Results [7 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 209982.500 	 std = 27363.471
Completed tasks: mean = 583.100 	 std = 7.106
Total cost:      mean = 6381.600 	 std = 8.452
Avg cost:        mean = 10.946 	 std = 0.135
Distribution:    mean = 3.512 	 std = 0.070
=====================================================

=====================================================
Experiment Results [8 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 321554.300 	 std = 23428.780
Completed tasks: mean = 656.200 	 std = 8.976
Total cost:      mean = 7305.800 	 std = 14.627
Avg cost:        mean = 11.136 	 std = 0.166
Distribution:    mean = 3.710 	 std = 0.157
=====================================================

=====================================================
Experiment Results [9 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 487888.900 	 std = 28434.127
Completed tasks: mean = 727.200 	 std = 8.807
Total cost:      mean = 8212.900 	 std = 17.598
Avg cost:        mean = 11.296 	 std = 0.145
Distribution:    mean = 3.898 	 std = 0.166
=====================================================

=====================================================
Experiment Results [10 agents] [5 grid-size] for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 678132.100 	 std = 46317.857
Completed tasks: mean = 784.700 	 std = 7.198
Total cost:      mean = 9037.500 	 std = 19.428
Avg cost:        mean = 11.518 	 std = 0.119
Distribution:    mean = 4.132 	 std = 0.122
=====================================================
"""


"""
A* Calls
| Algorithm                          | 1             | 2               | 3                | 4                 | 5                 | 6                  | 7                  | 8                  | 9                  | 10                   |
| ---------------------------------- | ------------- | --------------- | ---------------- | ----------------- | ----------------- | ------------------ | ------------------ | ------------------ | ------------------ | -------------------- |
| DECENTRALIZED_RESPECT              | 3627.2 (95.3) | 8616.4 (398.2)  | 14903.6 (515.1)  | 23603.7 (1281.8)  | 33813.4 (2082.8)  | 49391.7 (2447.0)   | 71365.0 (5322.2)   | 93859.8 (4993.3)   | 137436.5 (14004.0) | 192568.9 (17932.3)   |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 3627.2 (95.3) | 11827.1 (588.4) | 24001.0 (2077.5) | 57626.1 (19747.2) | 85014.5 (16216.9) | 159407.4 (36010.8) | 282301.2 (42488.5) | 460835.5 (74604.5) | 672043.0 (88455.7) | 1092802.2 (102441.0) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 3627.2 (95.3) | 10971.6 (554.2) | 22421.4 (1728.5) | 41284.0 (2358.1)  | 71787.9 (9656.9)  | 117762.1 (11983.3) | 190382.8 (14146.9) | 286685.2 (22163.8) | 428890.9 (36802.2) | 643188.6 (52284.9)   |
| DECENTRALIZED_NEGOTIATE_KARMA      | 3627.2 (95.3) | 10742.3 (837.8) | 24109.4 (1534.9) | 44633.4 (4453.4)  | 74146.1 (6520.7)  | 128263.7 (7622.1)  | 209982.5 (27363.5) | 321554.3 (23428.8) | 487888.9 (28434.1) | 678132.1 (46317.9)   |

Completed Tasks
| Algorithm                          | 1          | 2           | 3           | 4           | 5           | 6           | 7           | 8           | 9            | 10           |
| ---------------------------------- | ---------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ------------ | ------------ |
| DECENTRALIZED_RESPECT              | 85.2 (2.6) | 166.5 (3.8) | 247.7 (3.0) | 325.4 (6.1) | 401.9 (4.2) | 472.9 (6.5) | 538.5 (7.8) | 604.1 (9.7) | 654.6 (9.0)  | 701.7 (7.8)  |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 85.2 (2.6) | 168.0 (3.4) | 252.5 (2.8) | 340.1 (5.5) | 421.9 (4.6) | 505.4 (6.4) | 586.1 (6.7) | 657.8 (8.6) | 734.3 (5.4)  | 792.2 (7.4)  |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 85.2 (2.6) | 170.8 (3.2) | 256.9 (3.5) | 340.8 (6.3) | 426.9 (6.4) | 513.5 (4.7) | 595.0 (9.3) | 674.9 (6.1) | 752.4 (10.0) | 815.5 (11.7) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 85.2 (2.6) | 168.8 (3.9) | 253.3 (3.6) | 336.4 (4.3) | 417.0 (4.8) | 500.9 (5.4) | 583.1 (7.1) | 656.2 (9.0) | 727.2 (8.8)  | 784.7 (7.2)  |

Total Cost
| Algorithm                          | 1           | 2            | 3            | 4             | 5            | 6             | 7             | 8             | 9             | 10            |
| ---------------------------------- | ----------- | ------------ | ------------ | ------------- | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- |
| DECENTRALIZED_RESPECT              | 910.5 (4.5) | 1822.0 (7.9) | 2737.0 (6.0) | 3657.4 (7.3)  | 4575.6 (6.1) | 5496.5 (7.4)  | 6422.7 (16.8) | 7340.7 (11.2) | 8294.4 (18.7) | 9152.2 (23.6) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 910.5 (4.5) | 1822.2 (5.8) | 2734.6 (8.8) | 3646.4 (5.2)  | 4557.7 (5.8) | 5465.2 (14.6) | 6375.7 (10.9) | 7299.4 (13.5) | 8218.0 (16.6) | 9049.0 (26.1) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 910.5 (4.5) | 1818.9 (4.3) | 2732.0 (8.0) | 3636.1 (7.7)  | 4554.1 (7.6) | 5460.3 (7.5)  | 6371.5 (11.5) | 7284.6 (9.2)  | 8197.1 (11.0) | 9012.9 (25.2) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 910.5 (4.5) | 1821.6 (8.4) | 2733.5 (5.7) | 3644.9 (10.6) | 4560.9 (6.3) | 5468.0 (11.4) | 6381.6 (8.5)  | 7305.8 (14.6) | 8212.9 (17.6) | 9037.5 (19.4) |

Avg Cost
| Algorithm                          | 1            | 2            | 3            | 4            | 5            | 6            | 7            | 8            | 9            | 10           |
| ---------------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| DECENTRALIZED_RESPECT              | 10.70 (0.33) | 10.95 (0.27) | 11.05 (0.15) | 11.24 (0.22) | 11.39 (0.12) | 11.63 (0.17) | 11.93 (0.19) | 12.16 (0.19) | 12.67 (0.19) | 13.05 (0.17) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 10.70 (0.33) | 10.85 (0.24) | 10.83 (0.15) | 10.73 (0.19) | 10.80 (0.12) | 10.82 (0.16) | 10.88 (0.13) | 11.10 (0.16) | 11.19 (0.08) | 11.42 (0.12) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 10.70 (0.33) | 10.65 (0.22) | 10.64 (0.16) | 10.67 (0.20) | 10.67 (0.17) | 10.63 (0.10) | 10.71 (0.18) | 10.79 (0.10) | 10.90 (0.15) | 11.06 (0.18) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 10.70 (0.33) | 10.80 (0.28) | 10.79 (0.15) | 10.84 (0.15) | 10.94 (0.13) | 10.92 (0.13) | 10.95 (0.14) | 11.14 (0.17) | 11.30 (0.15) | 11.52 (0.12) |

Distribution
| Algorithm                          | 1           | 2           | 3           | 4           | 5           | 6           | 7           | 8           | 9           | 10          |
| ---------------------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| DECENTRALIZED_RESPECT              | 2.85 (0.18) | 2.81 (0.12) | 3.05 (0.08) | 3.21 (0.11) | 3.37 (0.08) | 3.48 (0.12) | 3.73 (0.11) | 3.94 (0.11) | 4.26 (0.10) | 4.46 (0.10) |
| DECENTRALIZED_NEGOTIATE_EGOISTIC   | 2.85 (0.18) | 2.75 (0.10) | 2.86 (0.13) | 2.98 (0.13) | 2.97 (0.11) | 3.09 (0.08) | 3.19 (0.10) | 3.27 (0.06) | 3.45 (0.11) | 3.60 (0.09) |
| DECENTRALIZED_NEGOTIATE_ALTRUISTIC | 2.85 (0.18) | 2.80 (0.18) | 2.91 (0.12) | 2.92 (0.11) | 3.04 (0.09) | 3.01 (0.10) | 3.19 (0.08) | 3.21 (0.13) | 3.29 (0.11) | 3.47 (0.17) |
| DECENTRALIZED_NEGOTIATE_KARMA      | 2.85 (0.18) | 2.77 (0.13) | 2.96 (0.16) | 3.07 (0.12) | 3.16 (0.14) | 3.31 (0.13) | 3.51 (0.07) | 3.71 (0.16) | 3.90 (0.17) | 4.13 (0.12) |
"""