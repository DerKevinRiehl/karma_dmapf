"""CONFLICT BASED SEARCH"""

import heapq


DIRS = [(0,1),(1,0),(0,-1),(-1,0)]  # N,E,S,W

class CBS_State:
    def __init__(self, x, y, theta, t, action):
        self.x = x
        self.y = y
        self.theta = theta
        self.t = t
        self.action = action

    def __repr__(self):
        return f"State(x={self.x}, y={self.y}, theta={self.theta}, t={self.t}, action={self.action})"
    
    def __lt__(self, other):
        return self.t < other.t

class CBS_Constraint:
    def __init__(self, agent, x, y, t):
        self.agent = agent
        self.x = x
        self.y = y
        self.t = t

class CBS_Node:
    def __init__(self):
        self.constraints = []
        self.paths = []
        self.cost = 0
        
class Planner_CBS:
    def __init__(self):
        pass
        
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
                    s = path[-1]
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
    
    def astar(self, grid, start, goal, agent, constraints):
        open_list = []
        visited = set()
        start_state = CBS_State(start[0], start[1], start[2], 0, "start")
        heapq.heappush(open_list, (0, start_state, []))
        while open_list:
            f, state, path = heapq.heappop(open_list)
            if (state.x, state.y, state.theta, state.t) in visited:
                continue
            visited.add((state.x, state.y, state.theta, state.t))
            new_path = path + [state]
            if (state.x, state.y) == goal:
                return new_path
            next_t = state.t + 1
            # -------------------------------------------------
            # actions
            # -------------------------------------------------
            # wait
            if not self.violates_constraint(agent, state.x, state.y, next_t, constraints):
                heapq.heappush(open_list, (
                    next_t + self.manhattan_heuristic((state.x,state.y), goal),
                    CBS_State(state.x, state.y, state.theta, next_t, "T"),
                    new_path
                ))
            # turn left (counterclockwise)
            heapq.heappush(open_list, (
                next_t + self.manhattan_heuristic((state.x,state.y), goal),
                CBS_State(state.x, state.y, (state.theta-1)%4, next_t, "A"),
                new_path
            ))
            # turn right
            heapq.heappush(open_list, (
                next_t + self.manhattan_heuristic((state.x,state.y), goal),
                CBS_State(state.x, state.y, (state.theta+1)%4, next_t, "C"),
                new_path
            ))
            # forward - convert direction to cardinal
            dx, dy = DIRS[state.theta]
            nx, ny = state.x+dx, state.y+dy
            direction_names = ["N", "E", "S", "W"]
            action_name = f"{direction_names[state.theta]}"
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx]==0:
                if not self.violates_constraint(agent, nx, ny, next_t, constraints):
                    heapq.heappush(open_list, (
                        next_t + self.manhattan_heuristic((nx,ny), goal),
                        CBS_State(nx, ny, state.theta, next_t, action_name),
                        new_path
                    ))    
        return None
    
    def plan_cbs(self, grid, starts, goals):
        root = CBS_Node()
        # initial paths
        for i,start in enumerate(starts):
            path = self.astar(grid,start,goals[i],i,root.constraints)
            if path is None:
                return None
            root.paths.append(path)
        root.cost = self.compute_cost(root.paths)
        # replanning
        open_list = []
        heapq.heappush(open_list,(root.cost,root))
        while open_list:
            _,node = heapq.heappop(open_list)
            conflict = self.detect_conflict(node.paths)
            if conflict is None:
                return node.paths
            for agent in [conflict["a1"],conflict["a2"]]:
                child = CBS_Node()
                child.constraints = list(node.constraints)
                x,y = conflict["pos"]
                t = conflict["time"]
                child.constraints.append(CBS_Constraint(agent,x,y,t))
                child.paths = list(node.paths)
                new_path = self.astar(
                    grid,
                    starts[agent],
                    goals[agent],
                    agent,
                    child.constraints
                )
                if new_path is None:
                    continue
                child.paths[agent] = new_path
                child.cost = self.compute_cost(child.paths)
                heapq.heappush(open_list,(child.cost,child))
        return None

    def convert_paths_to_routes(self, paths):
        routes = []
        for i,p in enumerate(paths):
            route = [s.action for s in p][1:]
            routes.append(route)
        return routes

    def plan(self, grid, starts, goals):
        paths = self.plan_cbs(grid, starts, goals)
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