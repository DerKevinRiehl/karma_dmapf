from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent
    from environment import Environment
    from planner_path_astar import PathPlannerState
    from numpy.typing import NDArray


from constants import (
    AGENT_ORIENTATION_EAST,
    AGENT_ORIENTATION_WEST,
    AGENT_ORIENTATION_NORTH,
    AGENT_ORIENTATION_SOUTH,
)

import numpy as np
from constants import SQUARE_SYMBOL_EMPTY, SQUARE_SYMBOL_OCCUPIED, SPAWN_BORDER, SPAWN_OCCUPIED_CELLS_BORDER


class GridTools:
    @staticmethod
    def create_dynamic_occupancy_grid(
        environment: Environment,
        time_horizon: int,
        agent_list: Optional[List[Agent]] = None,
        tabu_agent: Optional[Agent] = None,
    ) -> NDArray[np.bool_]:
        # For decentralized planning we want to consider just-vacated positions as occupied
        # for one additional timestep to prevent immediate re-entry. Enable that here.
        reservation_grid = GridTools.create_3D_reservation_grid(
            environment, time_horizon, agent_list, tabu_agent, consider_vacated_positions=True
        )
        dynamic_occupancy: NDArray[np.bool_] = reservation_grid != 0
        return dynamic_occupancy

    @staticmethod
    def create_3D_reservation_grid(
        environment: Environment,
        time_horizon: int,
        agent_list: Optional[List[Agent]] = None,
        tabu_agent: Optional[Agent] = None,
        consider_vacated_positions: bool = False,
    ) -> NDArray[np.int_]:
        """Create a 3D reservation table (time,x,y) with optional blocking of just-vacated cells.

        If consider_vacated_positions is True, whenever an agent occupies a cell at time t we also
        mark that same cell as occupied at time t+1 (if within the horizon). This prevents another
        agent from stepping into a cell immediately after it was vacated.
        """
        reservation_table: NDArray[np.int_] = np.zeros(
            (time_horizon + 1, environment.grid.grid_size, environment.grid.grid_size),
            dtype=int,
        )

        if agent_list is None:
            agent_list = environment.agents.copy()

        for agent in agent_list:
            if tabu_agent is not None and agent == tabu_agent:
                continue

            # init
            time_counter: int = 0
            current_pos: List[int] = agent.current_position.copy()

            # first position
            if len(agent.route) > 0:
                reservation_table[0][current_pos[0]][current_pos[1]] = agent.id
                if consider_vacated_positions and 1 < reservation_table.shape[0]:
                    reservation_table[1][current_pos[0]][current_pos[1]] = agent.id
            time_counter += 1

            # part of route
            for step in agent.route:
                if step == "C" or step == "A" or step == "T":
                    pass
                elif step == "N":
                    current_pos[1] += 1
                elif step == "S":
                    current_pos[1] -= 1
                elif step == "E":
                    current_pos[0] += 1
                elif step == "W":
                    current_pos[0] -= 1

                if time_counter < reservation_table.shape[0]:
                    reservation_table[time_counter][current_pos[0]][current_pos[1]] = agent.id
                    if consider_vacated_positions and (time_counter + 1) < reservation_table.shape[0]:
                        reservation_table[time_counter + 1][current_pos[0]][current_pos[1]] = agent.id
                time_counter += 1
                if time_counter == reservation_table.shape[0]:
                    break
            # end

            while time_counter < time_horizon:
                reservation_table[time_counter][current_pos[0]][current_pos[1]] = agent.id
                if consider_vacated_positions and (time_counter + 1) < reservation_table.shape[0]:
                    reservation_table[time_counter + 1][current_pos[0]][current_pos[1]] = agent.id
                time_counter += 1
        return reservation_table

    @staticmethod
    def detect_conflicts(
        path: List[PathPlannerState], reservation_table: NDArray[np.int_]
    ) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        conflicting_agents: List[int] = []
        for state in path:
            t: int = state.t
            if t >= reservation_table.shape[0]:
                break
            x: int = state.x
            y: int = state.y
            occupying_agent: int = int(reservation_table[t][x][y])
            if occupying_agent != 0:
                if occupying_agent not in conflicting_agents:
                    conflicts.append(
                        {
                            "time": t,
                            "position": (x, y),
                            "conflicting_agent": occupying_agent,
                        }
                    )
                    conflicting_agents.append(occupying_agent)
        return conflicts


class Grid:
    def __init__(self, grid_size: int):
        self.grid_size: int = grid_size
        self.occupancy_grid: NDArray[np.int_] = SQUARE_SYMBOL_EMPTY * np.ones(
            (grid_size, grid_size), dtype=int
        )

    def get_random_empty_square(self) -> Optional[List[int]]:
        empty_indices = np.argwhere(self.occupancy_grid == SQUARE_SYMBOL_EMPTY)
        if empty_indices.size == 0:
            return None
        idx: int = np.random.choice(len(empty_indices))
        x, y = empty_indices[idx]
        return [int(x), int(y)]

    @staticmethod
    def _occupy_with_border(
        grid: NDArray[np.int_], x: int, y: int, border: int, occupied_symbol: int
    ) -> None:
        max_x: int = len(grid)
        max_y: int = len(grid[0])
        for dx in range(-border, border + 1):
            for dy in range(-border, border + 1):
                nx: int = x + dx
                ny: int = y + dy
                if 0 <= nx < max_x and 0 <= ny < max_y:
                    grid[nx][ny] = occupied_symbol

    def get_random_empty_square_no_tasks(
        self, environment: Environment, pos: Optional[List[int]] = None
    ) -> Optional[List[int]]:
        temp_occupancy_grid: NDArray[np.int_] = self.occupancy_grid.copy()

        # other tasks
        for task in environment.tasks:
            Grid._occupy_with_border(
                temp_occupancy_grid,
                task.from_position[0],
                task.from_position[1],
                SPAWN_OCCUPIED_CELLS_BORDER,
                SQUARE_SYMBOL_OCCUPIED,
            )
            Grid._occupy_with_border(
                temp_occupancy_grid,
                task.to_position[0],
                task.to_position[1],
                SPAWN_OCCUPIED_CELLS_BORDER,
                SQUARE_SYMBOL_OCCUPIED,
            )

        # other agents
        for agent in environment.agents:
            Grid._occupy_with_border(
                temp_occupancy_grid,
                agent.current_position[0],
                agent.current_position[1],
                SPAWN_OCCUPIED_CELLS_BORDER,
                SQUARE_SYMBOL_OCCUPIED,
            )

        # additional points
        if pos is not None:
            Grid._occupy_with_border(
                temp_occupancy_grid,
                pos[0],
                pos[1],
                SPAWN_OCCUPIED_CELLS_BORDER,
                SQUARE_SYMBOL_OCCUPIED,
            )

        # spawn border
        for idx in range(0, SPAWN_BORDER):
            temp_occupancy_grid[idx, :] = SQUARE_SYMBOL_OCCUPIED
            temp_occupancy_grid[self.grid_size - 1 - idx, :] = SQUARE_SYMBOL_OCCUPIED
            temp_occupancy_grid[:, idx] = SQUARE_SYMBOL_OCCUPIED
            temp_occupancy_grid[:, self.grid_size - 1 - idx] = SQUARE_SYMBOL_OCCUPIED

        # determine empty spaces for candidates
        empty_indices = np.argwhere(temp_occupancy_grid == SQUARE_SYMBOL_EMPTY)
        if empty_indices.size == 0:
            return None

        idx = np.random.choice(len(empty_indices))
        x, y = empty_indices[idx]
        return [int(x), int(y)]

    def occupy(self, position: List[int]) -> None:
        self.occupancy_grid[position[0], position[1]] = SQUARE_SYMBOL_OCCUPIED

    def release(self, position: List[int]) -> None:
        self.occupancy_grid[position[0], position[1]] = SQUARE_SYMBOL_EMPTY


class Geometry:
    @staticmethod
    def mahattan_distance(
        position_a: Tuple[int, int], position_b: Tuple[int, int]
    ) -> int:
        a_x, a_y = position_a
        b_x, b_y = position_b
        return abs(a_x - b_x) + abs(a_y - b_y)

    @staticmethod
    def rotation_distance(start_orientation: int, required_orientation: int) -> int:
        """Minimum number of rotations between two orientations."""
        diff: int = abs(start_orientation - required_orientation)
        return min(diff, 4 - diff)

    @staticmethod
    def travel_time_with_rotation(
        position_a: Tuple[int, int], position_b: Tuple[int, int], start_orientation: int
    ) -> int:
        """Estimate travel time including rotation cost."""
        ax, ay = position_a
        bx, by = position_b
        dx: int = bx - ax
        dy: int = by - ay
        move_cost: int = abs(dx) + abs(dy)
        # if already there
        if move_cost == 0:
            return 0
        # determine required first movement direction
        if abs(dx) > abs(dy):
            if dx > 0:
                needed_orientation = AGENT_ORIENTATION_EAST
            else:
                needed_orientation = AGENT_ORIENTATION_WEST
        else:
            if dy > 0:
                needed_orientation = AGENT_ORIENTATION_NORTH
            else:
                needed_orientation = AGENT_ORIENTATION_SOUTH
        rotation_cost: int = Geometry.rotation_distance(
            start_orientation, needed_orientation
        )
        return move_cost + rotation_cost
