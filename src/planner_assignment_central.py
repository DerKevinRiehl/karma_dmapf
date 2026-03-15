import numpy as np
from geometry import Geometry
from scipy.optimize import linear_sum_assignment

class Planner_Assignment_Central:
    def plan_assignment(idle_agents, open_tasks, time, alpha=0.2):
        """Compute optimal agent-task assignment using Hungarian algorithm."""
        num_agents = len(idle_agents)
        num_tasks = len(open_tasks)
        cost_matrix = np.zeros((num_agents, num_tasks))
        for i, agent in enumerate(idle_agents):
            for j, task in enumerate(open_tasks):
                dist = Geometry.travel_time_with_rotation(
                    agent.current_position,
                    task.from_position,
                    agent.current_orientation
                )
                wait_time = time - task.spawned_time
                cost_matrix[i, j] = dist - alpha * wait_time
        # Hungarian algorithm
        agent_indices, task_indices = linear_sum_assignment(cost_matrix)
        return agent_indices, task_indices
    
