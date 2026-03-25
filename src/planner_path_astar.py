"""A STAR BASED PATH PLANNER"""

from __future__ import annotations
from typing import List, Optional, Tuple, Set, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from agent import Agent

import heapq
from constants import DIRS, DIR_NAMES


class PathPlannerState:
    def __init__(self, x: int, y: int, theta: int, t: int, action: Optional[str]):
        self.x: int = x
        self.y: int = y
        self.theta: int = theta
        self.t: int = t
        self.action: Optional[str] = action

    def __lt__(self, other: "PathPlannerState") -> bool:
        return self.t < other.t

    def __repr__(self) -> str:
        return f"PathPlannerState(x={self.x}, y={self.y}, theta={self.theta}, t={self.t}, action={self.action})"


class AStarPathPlanner:
    COUNTER: int = 0

    def __init__(
        self, static_occupancy_grid: NDArray[np.int_], astar_params: Dict[str, Any]
    ):
        self.static_occupancy_grid: NDArray[np.int_] = static_occupancy_grid
        self.astar_params: Dict[str, Any] = astar_params

    def manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(
        self,
        start: Tuple[int, int, int],
        goal: Tuple[int, int],
        dynamic_occupancy: Optional[NDArray[np.bool_]] = None,
        ignore_counter: bool = False,
    ) -> Optional[List[PathPlannerState]]:
        """
        This is the implementation of a astar algorithm for a robot that needs
        to rotate into the direction of travel, and can wait.
        It considers obstables from a static_occupancy and dynamic_occupancy map.

        start: (x, y, theta)
        goal: (x, y)
        dynamic_occupancy: 3D occupancy grid [time, x, y]
        max_time_horizon: optional maximum time to consider
        ignore_counter: if True, do not increment the AStarPathPlanner.COUNTER (used for shortest possible path calculation for evaluation only)
        """
        open_list: List[Tuple[int, PathPlannerState, List[PathPlannerState]]] = []
        visited: Set[Tuple[int, int, int, int]] = set()
        steps: int = 0
        start_state: PathPlannerState = PathPlannerState(
            start[0], start[1], start[2], 0, "start"
        )
        heapq.heappush(open_list, (0, start_state, []))
        while open_list:
            if not ignore_counter:
                AStarPathPlanner.COUNTER += 1

            steps += 1
            # ABORT CONDITION: TIMEOUT
            if steps > self.astar_params["max_iterations"]:
                # print("\t\tASTAR [TIMEOUT] A* aborted")
                return None
            # EXPLORE NEW STEP
            f, state, path = heapq.heappop(open_list)
            key: Tuple[int, int, int, int] = (state.x, state.y, state.theta, state.t)
            if key in visited:
                continue
            visited.add(key)
            new_path: List[PathPlannerState] = path + [state]

            # ABORT CONDITION: FOUND GOAL
            if (state.x, state.y) == goal:
                return new_path

            # EXPLORE
            next_t: int = state.t + 1
            if next_t > self.astar_params["planning_horizon"]:
                continue

            # BRANCH 1: ACTION: WAIT
            if (
                dynamic_occupancy is None
                or not dynamic_occupancy[next_t, state.x, state.y]
            ):
                heapq.heappush(
                    open_list,
                    (
                        next_t + self.manhattan((state.x, state.y), goal),
                        PathPlannerState(state.x, state.y, state.theta, next_t, "T"),
                        new_path,
                    ),
                )

            # BRANCH 2: ACTION: ROTATE (turn left/right)
            heapq.heappush(
                open_list,
                (
                    next_t + self.manhattan((state.x, state.y), goal),
                    PathPlannerState(
                        state.x, state.y, (state.theta - 1) % 4, next_t, "A"
                    ),
                    new_path,
                ),
            )
            heapq.heappush(
                open_list,
                (
                    next_t + self.manhattan((state.x, state.y), goal),
                    PathPlannerState(
                        state.x, state.y, (state.theta + 1) % 4, next_t, "C"
                    ),
                    new_path,
                ),
            )

            # BRANCH 3: ACTION: MOVE FORWARD
            dx, dy = DIRS[state.theta]
            nx, ny = state.x + dx, state.y + dy
            if (
                0 <= nx < self.static_occupancy_grid.shape[0]
                and 0 <= ny < self.static_occupancy_grid.shape[1]
            ):
                if self.static_occupancy_grid[nx, ny] == 0:
                    if (
                        dynamic_occupancy is None
                        or not dynamic_occupancy[next_t, nx, ny]
                    ):
                        heapq.heappush(
                            open_list,
                            (
                                next_t + self.manhattan((nx, ny), goal),
                                PathPlannerState(
                                    nx, ny, state.theta, next_t, DIR_NAMES[state.theta]
                                ),
                                new_path,
                            ),
                        )
        # print("\t\tASTAR [No valid path found in given time horizon]")
        return None

    def convert_path_to_route(
        self, path: Optional[List[PathPlannerState]]
    ) -> Optional[List[str]]:
        if path is None:
            return None
        return [s.action for s in path if s.action is not None][1:]  # skip start state

    def convert_route_to_path(self, agent: "Agent") -> Optional[List[PathPlannerState]]:
        """
        Reconstruct a path (list of PathPlannerState) from a start state and action list.

        start: (x, y, theta)
        actions: list of actions (e.g. ["T", "A", "C", "N", ...])
        """
        if agent.route is None:
            return None

        path: List[PathPlannerState] = []
        x: int = agent.current_position[0]
        y: int = agent.current_position[1]
        theta: int = agent.current_orientation
        t: int = 0

        # include start state
        path.append(PathPlannerState(x, y, theta, t, "start"))
        for action in agent.route:
            t += 1
            if action == "T":  # wait
                pass
            elif action == "A":  # rotate left
                theta = (theta - 1) % 4
            elif action == "C":  # rotate right
                theta = (theta + 1) % 4
            else:
                # assume forward movement (N/E/S/W from DIR_NAMES)
                dx, dy = DIRS[theta]
                x += dx
                y += dy
            path.append(PathPlannerState(x, y, theta, t, action))

        return path
