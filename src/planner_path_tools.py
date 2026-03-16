"""A STAR BASED PATH PLANNER"""

import heapq


DIRS = [(0,1),(1,0),(0,-1),(-1,0)]  # N,E,S,W
DIR_NAMES = ["N", "E", "S", "W"]

class PathPlannerState:
    def __init__(self, x, y, theta, t, action):
        self.x = x
        self.y = y
        self.theta = theta
        self.t = t
        self.action = action
    
    def __lt__(self, other):
        return self.t < other.t

    def __repr__(self):
        return f"PathPlannerState(x={self.x}, y={self.y}, theta={self.theta}, t={self.t}, action={self.action})"

class AStarPathPlanner:
    MAX_STEPS = 1000
    MAX_T = 100  # default maximum time for search

    def __init__(self, grid):
        self.grid = grid

    def manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def astar(self, start, goal, occupancy=None, max_t=None):
        """
        start: (x, y, theta)
        goal: (x, y)
        occupancy: 3D occupancy grid [time, x, y]
        max_t: optional maximum time to consider
        """
        if max_t is None:
            max_t = self.MAX_T

        open_list = []
        visited = set()
        steps = 0
        start_state = PathPlannerState(start[0], start[1], start[2], 0, "start")
        heapq.heappush(open_list, (0, start_state, []))

        while open_list:
            steps += 1
            if steps > self.MAX_STEPS:
                print("[TIMEOUT] A* aborted")
                return None

            f, state, path = heapq.heappop(open_list)
            key = (state.x, state.y, state.theta, state.t)
            if key in visited:
                continue
            visited.add(key)
            new_path = path + [state]

            if (state.x, state.y) == goal:
                return new_path

            next_t = state.t + 1
            if next_t > max_t:
                continue

            # Wait
            if occupancy is None or not occupancy[next_t, state.x, state.y]:
                heapq.heappush(open_list, (
                    next_t + self.manhattan((state.x, state.y), goal),
                    PathPlannerState(state.x, state.y, state.theta, next_t, "T"),
                    new_path
                ))

            # Turn left/right
            heapq.heappush(open_list, (
                next_t + self.manhattan((state.x, state.y), goal),
                PathPlannerState(state.x, state.y, (state.theta - 1) % 4, next_t, "A"),
                new_path
            ))
            heapq.heappush(open_list, (
                next_t + self.manhattan((state.x, state.y), goal),
                PathPlannerState(state.x, state.y, (state.theta + 1) % 4, next_t, "C"),
                new_path
            ))

            # Forward
            dx, dy = DIRS[state.theta]
            nx, ny = state.x + dx, state.y + dy
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                if self.grid[nx, ny] == 0:
                    if occupancy is None or not occupancy[next_t, nx, ny]:
                        heapq.heappush(open_list, (
                            next_t + self.manhattan((nx, ny), goal),
                            PathPlannerState(nx, ny, state.theta, next_t, DIR_NAMES[state.theta]),
                            new_path
                        ))
        return None

    def convert_path_to_actions(self, path):
        if path is None:
            return None
        return [s.action for s in path][1:]  # skip start
