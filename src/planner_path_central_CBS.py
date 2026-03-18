"""CONFLICT BASED SEARCH (CBS)"""

import heapq
import itertools
import numpy as np
from planner_path_tools import AStarPathPlanner

class CBS_Constraint:
    def __init__(self, agent, x, y, t):
        self.agent = agent
        self.x = x
        self.y = y
        self.t = t
    
    def __lt__(self, other):
        return self.t < other.t

class CBS_Node:
    def __init__(self):
        self.constraints = []
        self.paths = []
        self.cost = 0
    
    def __lt__(self, other):
        return self.cost < other.cost
        
class Planner_CBS:
    """
    This is the implementation of conflict based search (CBS).
    This algorithm plans paths using A* algorithm for multiple agents simultaneously
    that cannot cross each other, and need to consider each other when planning and
    executing their trajectory.
    
    Every branch (CBS_Node) of the search tree represents a set of different paths for all agents.
    The goal is find a set of paths (plan) that minimizes collective travel time.
    """
    MAX_T = 100
    MAX_CBS_NODES = 5000
    MAX_ASTAR_STEPS = 5000
    MAX_IDLE_TIME_CONSIDERED = 5
    def __init__(self, grid, astar_params):
        self.grid = grid
        self.astar_planner = AStarPathPlanner(grid, astar_params=astar_params)
    
    def detect_conflict(self, paths):
        max_len = max(len(p) for p in paths)
        for t in range(max_len):
            positions = {}
            for i,path in enumerate(paths):
                if t < len(path):
                    s = path[t]
                elif t<len(path)+Planner_CBS.MAX_IDLE_TIME_CONSIDERED:
                    s = path[-1]
                else:
                    continue # <- makes idle agent disappear
                    # s = path[-1]
                pos = (s.x,s.y)
                if pos in positions:
                    return {
                        "time":t,
                        "a1":positions[pos],
                        "a2":i,
                        "pos":pos
                    }
                positions[pos] = i
        return None
    
    def compute_cost(self, paths):
        return sum(len(p) for p in paths)

    def get_dynamic_occupancy_grid(self, constraints, agent):
        dynamic_occupancy = np.zeros((Planner_CBS.MAX_T + 1, self.grid.shape[0], self.grid.shape[1]), dtype=bool)
        for c in constraints:
            if c.agent == agent:
                dynamic_occupancy[c.t, c.x, c.y] = True
        return dynamic_occupancy
    
    def astar_launcher(self, start, goal, agent, constraints):
        # Build occupancy grid from constraints
        dynamic_occupancy = self.get_dynamic_occupancy_grid(constraints, agent)
        return self.astar_planner.astar(start=start, 
                                        goal=goal, 
                                        dynamic_occupancy=dynamic_occupancy, 
                                        max_time_horizon=Planner_CBS.MAX_T)
        
    def plan_cbs(self, starts, goals):
        root = CBS_Node()
        # initial paths
        for i, start in enumerate(starts):
            path = self.astar_launcher(start, goals[i], i, root.constraints)
            if path is None:
                return None
            root.paths.append(path)
        root.cost = self.compute_cost(root.paths)
        # branch and search, avoid conflicts, minize total costs
        open_list = []
        counter = itertools.count()
        heapq.heappush(open_list, (root.cost, next(counter), root))
        nodes_expanded = 0
        while open_list:
            _, _, node = heapq.heappop(open_list)
            nodes_expanded += 1
            # ABORT CONDITION: TIMEOUT
            if nodes_expanded > Planner_CBS.MAX_CBS_NODES:
                print("[TIMEOUT] aborted CBS")
                return None
            # Detect conflicts
            conflict = self.detect_conflict(node.paths)
            # Determine valid set of paths
            if conflict is None:
                return node.paths
            # Explore other paths for conflicts
            for agent in [conflict["a1"], conflict["a2"]]:
                child = CBS_Node()
                child.constraints = list(node.constraints)
                x, y = conflict["pos"]
                t = conflict["time"]
                child.constraints.append(CBS_Constraint(agent, x, y, t))
                child.paths = list(node.paths)
                start_state = child.paths[agent][0]  # PathPlannerState
                start = (start_state.x, start_state.y, start_state.theta)
                new_path = self.astar_launcher(start, goals[agent], agent, child.constraints)
                if new_path is None:
                    continue
                child.paths[agent] = new_path
                child.cost = self.compute_cost(child.paths)
                heapq.heappush(open_list, (child.cost, next(counter), child))
        return None

    def convert_paths_to_routes(self, paths):
        routes = []
        if paths is None:
            return None
        for i,p in enumerate(paths):
            route = [s.action for s in p][1:]
            routes.append(route)
        return routes

    def plan(self, starts, goals):
        paths = self.plan_cbs(starts, goals)
        return self.convert_paths_to_routes(paths)
