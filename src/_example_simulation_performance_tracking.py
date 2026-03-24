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
from visualization import plot_environment_and_reservation, make_gif

from constants import (
    MAPF_CONTROLLER_CENTRALIZED,
    MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA,
)

###############################################################################
###### PARAMETERS #############################################################
###############################################################################
simulation_settings = {
    "random_seed": 42,
    "grid_size": 14,
    "n_agents": 10,
    # "mapf_control": MAPF_CONTROLLER_CENTRALIZED,
    # "mapf_control": MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
    # "mapf_control": MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
    # "mapf_control": MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
    "mapf_control": MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA,
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


###############################################################################
###### MAIN ###################################################################
###############################################################################
astar_calls_list = []
n_completed_tasks_list = []
n_total_cost_list = []
n_average_cost_list = []
n_distribution_list = []

for random_seed in range(41, 51):
    simulation_settings["random_seed"] = random_seed
    n_astar_calls = 0
    print("Starting Experiment", random_seed)
    environment = Environment(settings=simulation_settings)
    # spawn initial agents
    for n in range(0, 10):  # int(N_AGENTS/2)):
        environment.spawn_agent()
    # spawn initial tasks
    for n in range(0, 10):  # int(N_AGENTS/2)):
        environment.spawn_task()
    # simulation loop
    while environment.time < environment.settings["time_simulation_duration"]:
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
        if len(environment.tasks) < len(environment.agents):
            environment.spawn_task()
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
    "Experiment Results for algorithm",
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
Experiment Results for algorithm DECENTRALIZED_RESPECT over 10 experiments
=====================================================
A* calls:        mean = 71275.800 	 std = 1886.910
Completed tasks: mean = 473.300 	 std = 5.515
Total cost:      mean = 9281.400 	 std = 37.479
Avg cost:        mean = 19.613 	 std = 0.269
Distribution:    mean = 5.756 	 std = 0.112
=====================================================

=====================================================
Experiment Results for algorithm DECENTRALIZED_NEGOTIATE_EGOISTIC over 10 experiments
=====================================================
A* calls:        mean = 134746.800 	 std = 6711.572
Completed tasks: mean = 486.000 	 std = 4.858
Total cost:      mean = 9263.400 	 std = 26.953
Avg cost:        mean = 19.062 	 std = 0.199
Distribution:    mean = 5.454 	 std = 0.137
=====================================================

=====================================================
Experiment Results for algorithm DECENTRALIZED_NEGOTIATE_ALTRUISTIC over 10 experiments
=====================================================
A* calls:        mean = 136193.900 	 std = 7432.091
Completed tasks: mean = 484.400 	 std = 4.543
Total cost:      mean = 9248.600 	 std = 29.971
Avg cost:        mean = 19.095 	 std = 0.222
Distribution:    mean = 5.649 	 std = 0.209
=====================================================

=====================================================
Experiment Results for algorithm DECENTRALIZED_NEGOTIATE_KARMA over 10 experiments
=====================================================
A* calls:        mean = 136949.600 	 std = 11524.341
Completed tasks: mean = 485.900 	 std = 5.430
Total cost:      mean = 9253.500 	 std = 24.258
Avg cost:        mean = 19.047 	 std = 0.226
Distribution:    mean = 5.584 	 std = 0.231
=====================================================
"""
