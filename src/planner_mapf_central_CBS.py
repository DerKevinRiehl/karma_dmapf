"""CONFLICT BASED SEARCH (CBS)"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING


import heapq
import itertools
import numpy as np
from planner_path_astar import AStarPathPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from planner_path_astar import PathPlannerState


class CBS_Constraint:
    def __init__(self, agent: int, x: int, y: int, t: int):
        self.agent: int = agent
        self.x: int = x
        self.y: int = y
        self.t: int = t

    def __lt__(self, other: CBS_Constraint) -> bool:
        return self.t < other.t


class CBS_Node:
    def __init__(self) -> None:
        self.constraints: List[CBS_Constraint] = []
        self.paths: List[List[PathPlannerState]] = []
        self.cost: int = 0

    def __lt__(self, other: CBS_Node) -> bool:
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

    def __init__(
        self,
        grid: NDArray[np.int_],
        cbs_params: Dict[str, Any],
        astar_params: Dict[str, Any],
    ):
        self.grid: NDArray[np.int_] = grid
        self.cbs_params: Dict[str, Any] = cbs_params
        self.astar_planner: AStarPathPlanner = AStarPathPlanner(
            grid, astar_params=astar_params
        )

    def detect_conflict(
        self, paths: List[List[PathPlannerState]]
    ) -> Optional[Dict[str, Any]]:
        max_len: int = max(len(p) for p in paths) if paths else 0

        for t in range(max_len):
            positions: Dict[Tuple[int, int], int] = {}
            for i, path in enumerate(paths):
                s: PathPlannerState
                if t < len(path):
                    s = path[t]
                elif t < len(path) + self.cbs_params["MAX_IDLE_TIME_CONSIDERED"]:
                    s = path[-1]
                else:
                    continue  # <- makes idle agent disappear
                    # s = path[-1]
                pos: Tuple[int, int] = (s.x, s.y)
                if pos in positions:
                    return {"time": t, "a1": positions[pos], "a2": i, "pos": pos}
                positions[pos] = i
        return None

    def compute_cost(self, paths: List[List[PathPlannerState]]) -> int:
        return sum(len(p) for p in paths)

    def get_dynamic_occupancy_grid(
        self, constraints: List[CBS_Constraint], agent: int
    ) -> NDArray[np.bool_]:
        dynamic_occupancy: NDArray[np.bool_] = np.zeros(
            (
                self.astar_planner.astar_params["planning_horizon"] + 1,
                self.grid.shape[0],
                self.grid.shape[1],
            ),
            dtype=bool,
        )
        for c in constraints:
            if c.agent == agent:
                dynamic_occupancy[c.t, c.x, c.y] = True
        return dynamic_occupancy

    def astar_launcher(
        self,
        start: Tuple[int, int, int],
        goal: Tuple[int, int],
        agent: int,
        constraints: List[CBS_Constraint],
    ) -> Optional[List[PathPlannerState]]:
        # Build occupancy grid from constraints
        dynamic_occupancy = self.get_dynamic_occupancy_grid(constraints, agent)
        return self.astar_planner.astar(
            start=start, goal=goal, dynamic_occupancy=dynamic_occupancy
        )

    def plan_cbs(
        self, starts: List[Tuple[int, int, int]], goals: List[Tuple[int, int]]
    ) -> Optional[List[List[PathPlannerState]]]:
        root: CBS_Node = CBS_Node()
        # initial paths
        for i, start in enumerate(starts):
            path = self.astar_launcher(start, goals[i], i, root.constraints)
            if path is None:
                return None
            root.paths.append(path)
        root.cost = self.compute_cost(root.paths)
        # branch and search, avoid conflicts, minize total costs
        open_list: List[Tuple[int, int, CBS_Node]] = []
        counter = itertools.count()
        heapq.heappush(open_list, (root.cost, next(counter), root))
        nodes_expanded: int = 0
        while open_list:
            _, _, node = heapq.heappop(open_list)
            nodes_expanded += 1
            # ABORT CONDITION: TIMEOUT
            if nodes_expanded > self.cbs_params["max_iterations"]:
                print("\t\t CBS [TIMEOUT]")
                return None
            # Detect conflicts
            conflict = self.detect_conflict(node.paths)
            # Determine valid set of paths
            if conflict is None:
                return node.paths
            # Explore other paths for conflicts
            for agent in [conflict["a1"], conflict["a2"]]:
                child: CBS_Node = CBS_Node()
                child.constraints = list(node.constraints)
                x, y = conflict["pos"]
                t: int = conflict["time"]
                child.constraints.append(CBS_Constraint(agent, x, y, t))
                child.paths = list(node.paths)
                start_state: PathPlannerState = child.paths[agent][0]
                start_pos: Tuple[int, int, int] = (
                    start_state.x,
                    start_state.y,
                    start_state.theta,
                )
                new_path = self.astar_launcher(
                    start_pos, goals[agent], agent, child.constraints
                )
                if new_path is None:
                    continue
                child.paths[agent] = new_path
                child.cost = self.compute_cost(child.paths)
                heapq.heappush(open_list, (child.cost, next(counter), child))
        print(
            "\t\tCBS [No conflict-free set of paths could be found in given time horizon]"
        )
        return None

    def convert_paths_to_routes(
        self, paths: Optional[List[List[PathPlannerState]]]
    ) -> Optional[List[List[str]]]:
        routes: List[List[str]] = []
        if paths is None:
            return None
        for i, p in enumerate(paths):
            route: List[str] = [s.action for s in p if s.action is not None][1:]
            routes.append(route)
        return routes

    def plan(
        self, starts: List[Tuple[int, int, int]], goals: List[Tuple[int, int]]
    ) -> Optional[List[List[str]]]:
        paths = self.plan_cbs(starts, goals)
        return self.convert_paths_to_routes(paths)
