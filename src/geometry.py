from constants import (
    AGENT_ORIENTATION_EAST,
    AGENT_ORIENTATION_WEST,
    AGENT_ORIENTATION_NORTH,
    AGENT_ORIENTATION_SOUTH,
)

import numpy as np
from constants import SQUARE_SYMBOL_EMPTY, SQUARE_SYMBOL_OCCUPIED, SPAWN_BORDER


class GridTools:
    def create_dynamic_occupancy_grid(
        environment, time_horizon, agent_list=None, tabu_agent=None
    ):
        reservation_grid = GridTools.create_3D_reservation_grid(
            environment, time_horizon, agent_list, tabu_agent
        )
        dynamic_occupancy = reservation_grid != 0
        return dynamic_occupancy

    def create_3D_reservation_grid(
        environment, time_horizon, agent_list=None, tabu_agent=None
    ):
        reservation_table = np.zeros(
            (time_horizon + 1, environment.grid.grid_size, environment.grid.grid_size)
        )
        if agent_list is None:
            agent_list = environment.agents.copy()
        for agent in agent_list:
            if tabu_agent is not None and agent == tabu_agent:
                continue
            # init
            time_counter = 0
            current_pos = agent.current_position.copy()
            # first position
            if len(agent.route) > 0:
                reservation_table[0][current_pos[0]][current_pos[1]] = agent.id
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
                reservation_table[time_counter][current_pos[0]][
                    current_pos[1]
                ] = agent.id
                time_counter += 1
                if time_counter == reservation_table.shape[0]:
                    break
            # end
            while time_counter < time_horizon:
                reservation_table[time_counter][current_pos[0]][
                    current_pos[1]
                ] = agent.id
                time_counter += 1
        return reservation_table

    def detect_conflicts(path, reservation_table):
        conflicts = []
        conflicting_agents = []
        for state in path:
            t = state.t
            if t >= reservation_table.shape[0]:
                break
            x = state.x
            y = state.y
            occupying_agent = reservation_table[t][x][y]
            if occupying_agent != 0:
                if occupying_agent not in conflicting_agents:
                    conflicts.append(
                        {
                            "time": t,
                            "position": (x, y),
                            "conflicting_agent": int(occupying_agent),
                        }
                    )
                    conflicting_agents.append(occupying_agent)
        return conflicts


class Grid:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.occupancy_grid = SQUARE_SYMBOL_EMPTY * np.ones((grid_size, grid_size))

    def get_random_empty_square(self):
        empty_indices = np.argwhere(self.occupancy_grid == SQUARE_SYMBOL_EMPTY)
        if empty_indices.size == 0:
            return None
        idx = np.random.choice(len(empty_indices))
        x, y = empty_indices[idx]
        return [int(x), int(y)]

    def _occupy_with_border(grid, x, y, border, occupied_symbol):
        max_x = len(grid)
        max_y = len(grid[0])
        for dx in range(-border, border + 1):
            for dy in range(-border, border + 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < max_x and 0 <= ny < max_y:
                    grid[nx][ny] = occupied_symbol

    def get_random_empty_square_no_tasks(self, environment, pos=None):
        temp_occupancy_grid = self.occupancy_grid.copy()
        # other tasks
        for task in environment.tasks:
            Grid._occupy_with_border(
                temp_occupancy_grid,
                task.from_position[0],
                task.from_position[1],
                SPAWN_BORDER,
                SQUARE_SYMBOL_OCCUPIED,
            )
            Grid._occupy_with_border(
                temp_occupancy_grid,
                task.to_position[0],
                task.to_position[1],
                SPAWN_BORDER,
                SQUARE_SYMBOL_OCCUPIED,
            )
        # other agents
        for agent in environment.agents:
            Grid._occupy_with_border(
                temp_occupancy_grid,
                agent.current_position[0],
                agent.current_position[1],
                SPAWN_BORDER,
                SQUARE_SYMBOL_OCCUPIED,
            )
        # additional points
        if pos is not None:
            Grid._occupy_with_border(
                temp_occupancy_grid,
                pos[0],
                pos[1],
                SPAWN_BORDER,
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

    def occupy(self, position):
        self.occupancy_grid[position[0], position[1]] = SQUARE_SYMBOL_OCCUPIED

    def release(self, position):
        self.occupancy_grid[position[0], position[1]] = SQUARE_SYMBOL_EMPTY


class Geometry:
    def mahattan_distance(position_a, position_b):
        a_x, a_y = position_a
        b_x, b_y = position_b
        return abs(a_x - b_x) + abs(a_y - b_y)

    def rotation_distance(start_orientation, required_orientation):
        """Minimum number of rotations between two orientations."""
        diff = abs(start_orientation - required_orientation)
        return min(diff, 4 - diff)

    def travel_time_with_rotation(position_a, position_b, start_orientation):
        """Estimate travel time including rotation cost."""
        ax, ay = position_a
        bx, by = position_b
        dx = bx - ax
        dy = by - ay
        move_cost = abs(dx) + abs(dy)
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
        rotation_cost = Geometry.rotation_distance(
            start_orientation, needed_orientation
        )
        return move_cost + rotation_cost
