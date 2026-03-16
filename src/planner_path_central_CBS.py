"""CONFLICT BASED SEARCH (CBS)"""

import heapq
import itertools
import numpy as np
from planner_path_tools import AStarPathPlanner

DIRS = [(0,1),(1,0),(0,-1),(-1,0)]  # N,E,S,W
DIR_NAMES = ["N", "E", "S", "W"]

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
    MAX_T = 100
    MAX_CBS_NODES = 1000
    MAX_ASTAR_STEPS = 1000
    
    def __init__(self, grid):
        self.grid = grid
        self.astar_planner = AStarPathPlanner(grid)
        
    def manhattan_heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    
    def violates_constraint(self, agent, x, y, t, constraints):
        for c in constraints:
            if c.agent == agent and c.x == x and c.y == y and c.t == t:
                return True
        return False
    
    def detect_conflict(self, paths):
        max_len = max(len(p) for p in paths)
        for t in range(max_len):
            positions = {}
            for i,path in enumerate(paths):
                if t < len(path):
                    s = path[t]
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

    def astar(self, start, goal, agent, constraints):
        # Build occupancy grid from constraints
        occupancy = np.zeros((Planner_CBS.MAX_T + 1, self.grid.shape[0], self.grid.shape[1]), dtype=bool)
        for c in constraints:
            if c.agent == agent:
                occupancy[c.t, c.x, c.y] = True
        return self.astar_planner.astar(start, goal, occupancy=occupancy, max_t=Planner_CBS.MAX_T)
    
    def plan_cbs(self, starts, goals):
        root = CBS_Node()
        # initial paths
        for i, start in enumerate(starts):
            path = self.astar(start, goals[i], i, root.constraints)
            if path is None:
                return None
            root.paths.append(path)
        root.cost = self.compute_cost(root.paths)
    
        open_list = []
        counter = itertools.count()
        heapq.heappush(open_list, (root.cost, next(counter), root))
        nodes_expanded = 0
    
        while open_list:
            _, _, node = heapq.heappop(open_list)
            nodes_expanded += 1
            if nodes_expanded > Planner_CBS.MAX_CBS_NODES:
                print("[TIMEOUT] aborted CBS")
                return None
    
            conflict = self.detect_conflict(node.paths)
            if conflict is None:
                return node.paths
    
            for agent in [conflict["a1"], conflict["a2"]]:
                child = CBS_Node()
                child.constraints = list(node.constraints)
                x, y = conflict["pos"]
                t = conflict["time"]
                child.constraints.append(CBS_Constraint(agent, x, y, t))
                child.paths = list(node.paths)
                start_state = child.paths[agent][0]  # PathPlannerState
                start = (start_state.x, start_state.y, start_state.theta)
                new_path = self.astar(start, goals[agent], agent, child.constraints)
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

"""
# -------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------
       
grid = [
    [0,0,0,0],
    [0,1,0,0],
    [0,0,0,0]
]

starts = [
    (0,0,1),  # x,y,theta
    (3,2,3)
]

goals = [
    (3,2),
    (0,2)
]


planner = Planner_CBS()
paths = planner.plan_cbs(grid,starts,goals)

for i,p in enumerate(paths):
    print("agent",i)
    for s in p:
        print(s)

routes = planner.convert_paths_to_routes(paths)

print(routes)
"""