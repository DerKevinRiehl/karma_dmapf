"""CONFLICT BASED SEARCH"""

import heapq
from collections import defaultdict, namedtuple

# -------------------------------------------------
# Basic definitions
# -------------------------------------------------

DIRS = [(0,-1),(1,0),(0,1),(-1,0)]  # N,E,S,W

State = namedtuple("State", ["x","y","theta","t", "action"])

class Constraint:
    def __init__(self, agent, x, y, t):
        self.agent = agent
        self.x = x
        self.y = y
        self.t = t


# -------------------------------------------------
# Low-level A* with orientation
# -------------------------------------------------

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def violates_constraint(agent, x, y, t, constraints):
    for c in constraints:
        if c.agent == agent and c.x == x and c.y == y and c.t == t:
            return True
    return False


def astar(grid, start, goal, agent, constraints):

    open_list = []
    visited = set()

    start_state = State(start[0], start[1], start[2], 0, "start")

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
        if not violates_constraint(agent, state.x, state.y, next_t, constraints):
            heapq.heappush(open_list, (
                next_t + heuristic((state.x,state.y), goal),
                State(state.x, state.y, state.theta, next_t, "wait"),
                new_path
            ))

        # turn left (counterclockwise)
        heapq.heappush(open_list, (
            next_t + heuristic((state.x,state.y), goal),
            State(state.x, state.y, (state.theta-1)%4, next_t, "rotate counterclockwise"),
            new_path
        ))

        # turn right
        heapq.heappush(open_list, (
            next_t + heuristic((state.x,state.y), goal),
            State(state.x, state.y, (state.theta+1)%4, next_t, "rotate clockwise"),
            new_path
        ))

        # forward - convert direction to cardinal
        dx, dy = DIRS[state.theta]
        nx, ny = state.x+dx, state.y+dy
        direction_names = ["north", "east", "south", "west"]
        action_name = f"{direction_names[state.theta]}"
        
        if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx]==0:
            if not violates_constraint(agent, nx, ny, next_t, constraints):
                heapq.heappush(open_list, (
                    next_t + heuristic((nx,ny), goal),
                    State(nx, ny, state.theta, next_t, action_name),
                    new_path
                ))

    return None

def detect_conflict(paths):

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

class CBSNode:

    def __init__(self):
        self.constraints = []
        self.paths = []
        self.cost = 0


def compute_cost(paths):
    return sum(len(p) for p in paths)


def cbs(grid, starts, goals):

    root = CBSNode()

    # initial paths
    for i,start in enumerate(starts):

        path = astar(grid,start,goals[i],i,root.constraints)

        if path is None:
            return None

        root.paths.append(path)

    root.cost = compute_cost(root.paths)

    open_list = []
    heapq.heappush(open_list,(root.cost,root))

    while open_list:

        _,node = heapq.heappop(open_list)

        conflict = detect_conflict(node.paths)

        if conflict is None:
            return node.paths

        for agent in [conflict["a1"],conflict["a2"]]:

            child = CBSNode()

            child.constraints = list(node.constraints)

            x,y = conflict["pos"]
            t = conflict["time"]

            child.constraints.append(Constraint(agent,x,y,t))

            child.paths = list(node.paths)

            new_path = astar(
                grid,
                starts[agent],
                goals[agent],
                agent,
                child.constraints
            )

            if new_path is None:
                continue

            child.paths[agent] = new_path
            child.cost = compute_cost(child.paths)

            heapq.heappush(open_list,(child.cost,child))

    return None


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

paths = cbs(grid,starts,goals)

for i,p in enumerate(paths):
    print("agent",i)
    for s in p:
        print(s)