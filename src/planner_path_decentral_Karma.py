"""CONFLICT BASED SEARCH (CBS)"""

import numpy as np
from planner_path_tools import AStarPathPlanner


class DecentralizedMAPF:
    def __init__(self, grid, max_time=50):
        self.grid = grid
        self.num_agents = 0
        self.max_time = max_time
        self.occupancy = np.zeros((self.max_time, grid.shape[0], grid.shape[1]), dtype=bool)
        self.planner = AStarPathPlanner(grid)

    def plan_paths(self, starts, goals):
        self.num_agents = len(starts)
        all_paths = []

        for agent_id in range(self.num_agents):
            start = starts[agent_id]
            goal = goals[agent_id]

            path = self.planner.astar(start, goal, occupancy=self.occupancy)
            if path is None:
                print(f"[Agent {agent_id}] Failed to find path!")
                all_paths.append(None)
                continue

            # Reserve cells in occupancy grid
            for state in path:
                if state.t < self.max_time:
                    self.occupancy[state.t, state.x, state.y] = True

            all_paths.append(path)

        return all_paths

    def convert_paths_to_actions(self, paths):
        return [self.planner.convert_path_to_actions(p) for p in paths]
