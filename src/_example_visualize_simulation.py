"""
POTENTIAL TITLE:
    KARMA MECHANISMS FOR DECENTRALIZED, ORIENTATION-AWARE MAPF

interesting repo: https://github.com/GavinPHR/Multi-Agent-Path-Finding?tab=readme-ov-file
"""

###############################################################################
###### IMPORTS ################################################################
###############################################################################
import os
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
    "grid_size": 5 + 2,  # 15,
    "n_agents": 10,
    # "mapf_control": MAPF_CONTROLLER_CENTRALIZED,
    # "mapf_control": MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
    # "mapf_control": MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
    # "mapf_control": MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
    "mapf_control": MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA,
    "time_horizon_visualization": 10,
    "time_simulation_duration": 100,
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
        "karma_influence": 0.2,
    },
    "debug_statements": True,
}


###############################################################################
###### MAIN ###################################################################
###############################################################################
environment = Environment(settings=simulation_settings)

# spawn initial agents
for n in range(0, simulation_settings["n_agents"]):
    environment.spawn_agent()

# simulation loop
SIMULATION_TIME_STEPS_STOP_SPAWNING = 70
while environment.time < environment.settings["time_simulation_duration"]:
    print(
        "\n\ntime:",
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
    if environment.time < SIMULATION_TIME_STEPS_STOP_SPAWNING:
        if len(environment.tasks) < len(environment.agents):
            environment.spawn_task()
        else:
            if np.random.random() > 0.9:
                if len(environment.tasks) * 2 + len(environment.agents) < 100 - 30:
                    environment.spawn_task()

    # handle tasks
    environment.assign_open_tasks()
    closed = environment.close_finished_tasks()

    # visualize
    os.makedirs("figs", exist_ok=True)
    plot_environment_and_reservation(
        environment, save_filename=f"figs/x_image_{environment.time:04d}.png"
    )

    # report A-STAR Calls
    print("\tA-Star Calls:", AStarPathPlanner.COUNTER)
    AStarPathPlanner.COUNTER = 0

os.makedirs("results", exist_ok=True)
make_gif(
    input_pattern="figs/x_image_*.png",
    output_gif=f"results/animation_{environment.settings['mapf_control']}.gif",
    duration=0.2,
)
