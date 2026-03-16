from constants import AGENT_ORIENTATION_EAST, AGENT_ORIENTATION_WEST, AGENT_ORIENTATION_NORTH, AGENT_ORIENTATION_SOUTH

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
            start_orientation,
            needed_orientation
        )
        return move_cost + rotation_cost