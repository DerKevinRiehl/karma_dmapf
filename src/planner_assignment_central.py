import numpy as np
from geometry import Geometry
from scipy.optimize import linear_sum_assignment
from constants import AGENT_STATUS_CARRY

class Planner_Assignment_Central:
    @staticmethod
    def estimate_agent_availability(agent, current_time):
        """
        Returns:
            (available_position, available_time, orientation)
        """
        # Case 1: idle agent
        if agent.is_idle():
            return agent.current_position, current_time, agent.current_orientation
        # Case 2: busy agent → assume it will finish current task (CARRYING STATUS)
        if agent.status == AGENT_STATUS_CARRY:
            remaining_steps = len(agent.route)
            # fallback: if no route exists yet, assume 0
            if remaining_steps is None:
                return agent.current_position, current_time, agent.current_orientation
            available_time = current_time + remaining_steps
            # after finishing, agent will be at delivery location
            available_position = agent.assigned_task.to_position
            # orientation approximation (keep current)
            orientation = agent.current_orientation
            return available_position, available_time, orientation
        
    @staticmethod
    def plan_assignment(candidate_agents, open_tasks, time, alpha=0.2):
        """
        Compute optimal agent-task assignment using Hungarian algorithm.
        The assignment costs include travel time (manhattan distance + rotation)
        but also the time since a task was not assigned yet.
        This tradeoff is balanced using a factor alpha.
        Agents that are in status IDLE or CARRY will be considered for assignment.
        
        all_agents: list of agents
        open_tasks: list of tasks
        time: current time
        alpha: weight for task's wait time in cost function
        """
        num_agents = len(candidate_agents)
        num_tasks = len(open_tasks)
        if num_agents == 0 or num_tasks == 0:
            return [], []
        cost_matrix = np.zeros((num_agents, num_tasks))
        for i, agent in enumerate(candidate_agents):
            start_pos, start_time, orientation = Planner_Assignment_Central.estimate_agent_availability(agent, time)
            for j, task in enumerate(open_tasks):
                # travel from availability position to pickup
                travel_time = Geometry.travel_time_with_rotation(
                    start_pos,
                    task.from_position,
                    orientation
                )
                arrival_time = start_time + travel_time
                wait_time = arrival_time - task.spawned_time
                cost_matrix[i, j] = arrival_time - alpha * wait_time
        # Hungarian algorithm
        agent_indices, task_indices = linear_sum_assignment(cost_matrix)
        return agent_indices, task_indices
