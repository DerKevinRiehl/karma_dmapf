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

grid_sizes = [5, 10, 15, 20]

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
                "params_karma": {"initial_karma": 5},
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