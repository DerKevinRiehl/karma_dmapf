"""
POTENTIAL TITLE: 
    KARMA MECHANISMS FOR DECENTRALIZED, ORIENTATION-AWARE MAPF
    
interesting repo: https://github.com/GavinPHR/Multi-Agent-Path-Finding?tab=readme-ov-file
"""

###############################################################################
###### IMPORTS ################################################################
###############################################################################
import numpy as np
from constants import SQUARE_SYMBOL_EMPTY, SQUARE_SYMBOL_OCCUPIED, AGENT_ORIENTATIONS, AGENT_STATUS_CARRY, AGENT_STATUS_PICKUP, AGENT_STATUS_IDLE
from constants import AGENT_ORIENTATION_SOUTH, AGENT_ORIENTATION_NORTH, AGENT_ORIENTATION_EAST, AGENT_ORIENTATION_WEST
from constants import ROUTE_CONTROLLER_CENTRALIZED, ROUTE_CONTROLLER_DECENTRALIZED_RESPECT, ROUTE_CONTROLLER_DECENTRALIZED_NEGOTIATE_MYOPIC, ROUTE_CONTROLLER_DECENTRALIZED_NEGOTIATE_KAMRA
from visualization import plot_environment_and_reservation, make_gif
from planner_path_central_CBS import Planner_CBS
from planner_assignment_central import Planner_Assignment_Central
from planner_path_tools import AStarPathPlanner


###############################################################################
###### PARAMETERS #############################################################
###############################################################################

SPAWN_BORDER = 1
GRID_SIZE = 14
N_AGENTS = 10
VISUALIZATION_TIME_HORIZON = 10
PLANNING_TIME_HORIZON = 20
SIMULATION_TIME_STEPS = 50
INITIAL_KARMA = 5
SELECTED_ROUTE_CONTROL = ROUTE_CONTROLLER_DECENTRALIZED_RESPECT # ROUTE_CONTROLLER_DECENTRALIZED_RESPECT
# SELECTED_ROUTE_CONTROL = ROUTE_CONTROLLER_CENTRALIZED

# central
astar_params = {
    "MAX_STEPS": 1000,
    "MAX_TIME_HORIZON": PLANNING_TIME_HORIZON
}

# decentral
astar_params = {
    "MAX_STEPS": 5000,
    "MAX_TIME_HORIZON": PLANNING_TIME_HORIZON
}


###############################################################################
###### CLASSES ################################################################
###############################################################################

class Agent:
    def __init__(self, agent_id, environment, is_karma_agent=False):
        self.id = agent_id
        self.environment = environment
        self.grid = self.environment.grid
        self.current_position = self.grid.get_random_empty_square()
        self.current_orientation = np.random.choice(AGENT_ORIENTATIONS)
        self.assigned_task = None
        self.status = AGENT_STATUS_IDLE
        self.route = []
        self.target_position = []
        self.grid.occupy(self.current_position)
        self.init_karma_specific_properties()
            
    def init_karma_specific_properties(self):
        self.karma_balance = INITIAL_KARMA
        self.path_planner = AStarPathPlanner(static_occupancy_grid=self.environment.static_grid.occupancy_grid, astar_params=astar_params)
        
    def is_idle(self):
        return self.assigned_task is None
    
    def is_available_soon(self):
        return self.status==AGENT_STATUS_CARRY
    
    def release_task(self):
        self.assigned_task = None
        self.status = AGENT_STATUS_IDLE
        self.target_position = []
        self.route = []
        
    def assign_task(self, task):
        self.assigned_task = task
        self.target_position = task.from_position
        if self.current_position==self.target_position:
            self.status = AGENT_STATUS_CARRY
        else:
            self.status = AGENT_STATUS_PICKUP

    def update_target_position(self):
        # determine target
        if self.status==AGENT_STATUS_PICKUP:
            self.target_position = self.assigned_task.from_position
        elif self.status==AGENT_STATUS_CARRY:
            self.target_position = self.assigned_task.to_position
        # update status in case hit
        if self.current_position==self.target_position:
            self.status = AGENT_STATUS_CARRY
            self.target_position = self.assigned_task.to_position
                
    def execute_route(self):
        if len(self.route)>0:
            task = self.route[0]
            self.route = self.route[1:]
            if task=="N":
                self._move_north()
            if task=="E":
                self._move_east()
            if task=="S":
                self._move_south()
            if task=="W":
                self._move_west()
            if task=="C":
                self._rotate_clockwise()
            if task=="A":
                self._rotate_counter_clockwise()
            if task=="T":
                pass
        
            if self.status == AGENT_STATUS_CARRY:
                self.assigned_task.current_position = self.current_position

    def _move_north(self):
        if self.current_position[1] < self.grid.grid_size-1 and self.current_orientation==AGENT_ORIENTATION_NORTH:
            self.grid.release(self.current_position)
            self.current_position[1] += 1
            self.grid.occupy(self.current_position)

    def _move_east(self):
        if self.current_position[0] < self.grid.grid_size-1 and self.current_orientation==AGENT_ORIENTATION_EAST:
            self.grid.release(self.current_position)
            self.current_position[0] += 1
            self.grid.occupy(self.current_position)     

    def _move_south(self):
        if self.current_position[1] > 0 and self.current_orientation==AGENT_ORIENTATION_SOUTH:
            self.grid.release(self.current_position)
            self.current_position[1] -= 1
            self.grid.occupy(self.current_position)     

    def _move_west(self): 
        if self.current_position[0] > 0 and self.current_orientation==AGENT_ORIENTATION_WEST:
            self.grid.release(self.current_position)
            self.current_position[0] -= 1
            self.grid.occupy(self.current_position)     

    def _rotate_clockwise(self):
        self.current_orientation += 1
        if self.current_orientation>AGENT_ORIENTATION_WEST:
            self.current_orientation = AGENT_ORIENTATION_NORTH

    def _rotate_counter_clockwise(self):
        self.current_orientation -= 1
        if self.current_orientation<AGENT_ORIENTATION_NORTH:
            self.current_orientation = AGENT_ORIENTATION_WEST

class Grid:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.occupancy_grid = SQUARE_SYMBOL_EMPTY*np.ones((grid_size, grid_size))

    def get_random_square(self):
        x = np.random.randint(0, self.grid_size)
        y = np.random.randint(0, self.grid_size)
        return [int(x), int(y)]

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
            Grid._occupy_with_border(temp_occupancy_grid, task.from_position[0], task.from_position[1], SPAWN_BORDER, SQUARE_SYMBOL_OCCUPIED)
            Grid._occupy_with_border(temp_occupancy_grid, task.to_position[0], task.to_position[1], SPAWN_BORDER, SQUARE_SYMBOL_OCCUPIED)
        # other agents
        for agent in environment.agents:
            Grid._occupy_with_border(temp_occupancy_grid, agent.current_position[0], agent.current_position[1], SPAWN_BORDER, SQUARE_SYMBOL_OCCUPIED)
        # additional points
        if pos is not None:
            Grid._occupy_with_border(temp_occupancy_grid, pos[0], pos[1], SPAWN_BORDER, SQUARE_SYMBOL_OCCUPIED)
        # spawn border
        for idx in range(0, SPAWN_BORDER):
            temp_occupancy_grid[idx,:] = SQUARE_SYMBOL_OCCUPIED
            temp_occupancy_grid[GRID_SIZE-1-idx,:] = SQUARE_SYMBOL_OCCUPIED
            temp_occupancy_grid[:,idx] = SQUARE_SYMBOL_OCCUPIED
            temp_occupancy_grid[:,GRID_SIZE-1-idx] = SQUARE_SYMBOL_OCCUPIED
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



class Task:
    def __init__(self, environment, task_id, grid, time):
        self.id = task_id
        self.grid = grid
        # this is to make sure that no origin or destination is set to the origin or destination of any other task or current position of robot
        # this facilitate solving path planning and less often gets aborted
        self.from_position = grid.get_random_empty_square_no_tasks(environment)
        self.to_position = grid.get_random_empty_square_no_tasks(environment, pos=self.from_position)
        if self.from_position is None or self.to_position is None:
            raise Exception()
        self.current_position = self.from_position.copy()
        #######################################################################
        self.assigned_agent = None
        self.spawned_time = time
        self.completed_time = None
 
    def is_assigned(self):
        return self.assigned_agent is not None

    def is_finished(self):
        return self.current_position==self.to_position and self.assigned_agent is not None


    
class Environment:
    def __init__(self, grid_size, route_control):
        self.grid = Grid(grid_size=grid_size)
        self.static_grid = Grid(grid_size=grid_size)
        self.time = 0
        self.agents = []
        self.tasks = []
        self.completed_tasks = []
        self.route_control = route_control
        
    def determine_new_id(self, lst):
        last_agent_id = 0
        if len(lst)>0:
            last_agent_id = lst[-1].id
        return last_agent_id + 1
    
    def spawn_agent(self):
        self.agents.append(
            Agent(
                agent_id=self.determine_new_id(self.agents), 
                environment=self
            )
        )
        
    def spawn_task(self):
        try:
            self.tasks.append(
                Task(
                    environment=self,
                    task_id=self.determine_new_id(self.tasks), 
                    grid=self.grid,
                    time=self.time
                )
            )
        except:
            # print("\ttoo crowded, cant spawn right now")
            pass
            
    def assign_open_tasks(self):
        candidate_agents = [a for a in self.agents if a.is_idle() or a.is_available_soon()]
        open_tasks = [t for t in self.tasks if not t.is_assigned()]
        if not candidate_agents or not open_tasks:
            return        
        agent_indices, task_indices = Planner_Assignment_Central.plan_assignment(candidate_agents, open_tasks, self.time)
        for a_idx, t_idx in zip(agent_indices, task_indices):
            agent = candidate_agents[a_idx]
            task = open_tasks[t_idx]
            # only assign if agent is idle (otherwise it is done in later iteration)
            if agent.is_idle():
                agent.assign_task(task)
                task.assigned_agent = agent
            
    def close_finished_tasks(self):
        finished_tasks = [task for task in self.tasks if task.is_finished()]
        for task in finished_tasks:
            task.assigned_agent.release_task()
            self.tasks.remove(task)
            task.completed_time = self.time
            self.completed_tasks.append(task)
        return len(finished_tasks)>0

    def handle_agents(self):
        # ROUTE EXECTUION: update agent target and status
        self.handle_agents_route_execution()
        # ROUTE PLANNING: for those who need
        if self.route_control==ROUTE_CONTROLLER_CENTRALIZED:
            self.handle_agents_route_planning_centralized()
        elif self.route_control==ROUTE_CONTROLLER_DECENTRALIZED_RESPECT:
            self.handle_agents_route_planning_decentralized_respect()
    
    def handle_agents_route_execution(self):
        for agent in self.agents:
            agent.execute_route()
            if len(agent.route)==0 and not agent.is_idle():
                agent.update_target_position()
    
    def handle_agents_route_planning_centralized(self):
        
        print("")
        print("Open Routes:")
        planning_relevant_agents = [agent for agent in self.agents if len(agent.route)>0]
        for agent in planning_relevant_agents:
            print("\t",agent.id, agent.route)
        print("")
        print("Current Agent Positions:")
        for agent in self.agents:
            print("\t",agent.id, "\t", agent.current_position, "\t| ", agent.target_position)
        print("")
        print("")
            
        # conduct centralized planning for running agents with jobs
        planning_relevant_agents = [agent for agent in self.agents if not agent.is_idle()]
        if len(planning_relevant_agents)>0:
            grid = self.grid.occupancy_grid*0
            starts = [(agent.current_position[0], agent.current_position[1], agent.current_orientation) for agent in planning_relevant_agents]
            goals = [(agent.target_position[0], agent.target_position[1]) for agent in planning_relevant_agents]
            planner = Planner_CBS(grid, astar_params=astar_params)
            # print("\tInput for Planner:", starts, goals, "\n", grid)
            routes = planner.plan(starts, goals)
            # update routes
            if routes is not None:
                for idx, agent in enumerate(planning_relevant_agents):
                    agent.route = routes[idx]

    def handle_agents_route_planning_decentralized_respect(self):
        """
        This works as follows: every new agent will plan its route around already existing planned routes, no negotiation, just adaption to others.
        """
        print("")
        print("Open Routes:")
        planning_relevant_agents = [agent for agent in self.agents if len(agent.route)>0]
        for agent in planning_relevant_agents:
            print("\t",agent.id, agent.route)
        dynamic_occupancy_grid = self.create_dynamic_occupancy_grid(time_horizon=PLANNING_TIME_HORIZON, agent_list=self.agents)
        print("")
        print("Current Agent Positions:")
        for agent in self.agents:
            print("\t",agent.id, "\t", agent.current_position, "\t| ", agent.target_position)
        print("")
        print("")
        
        # conduct decentralized planning for running agents with jobs who finished their current route
        planning_relevant_agents = [agent for agent in self.agents if (not agent.is_idle()) and len(agent.route)==0 and len(agent.target_position)==2]
        for agent in planning_relevant_agents:
            # determine dynamic_occupancy_grid given already planned routes
            dynamic_occupancy_grid = self.create_dynamic_occupancy_grid(time_horizon=PLANNING_TIME_HORIZON, agent_list=self.agents, tabu_agent=agent)
            path = agent.path_planner.astar(
                start=(agent.current_position[0], agent.current_position[1], agent.current_orientation), 
                goal=(agent.target_position[0], agent.target_position[1]), 
                dynamic_occupancy=dynamic_occupancy_grid)
            if path is not None:
                route = agent.path_planner.convert_path_to_actions(path)
                print("\trouteadded for ", agent.id, route)
                if  len(path)==0:
                    print("ERROR")
                    import sys
                    sys.exit(0)
                agent.route = route
            else:
                print("\tdas is der yarak: ",agent.id)
            
    def create_dynamic_occupancy_grid(self, time_horizon, agent_list=None, tabu_agent=None):
        reservation_grid = self.create_3D_reservation_grid(time_horizon, agent_list, tabu_agent)
        dynamic_occupancy = reservation_grid != 0
        return dynamic_occupancy
    
    def create_3D_reservation_grid(self, time_horizon, agent_list=None, tabu_agent=None):
        reservation_table = np.zeros((time_horizon+1, self.grid.grid_size, self.grid.grid_size))
        if agent_list is None:
            agent_list = self.agents.copy()
        for agent in agent_list:
            if tabu_agent is not None and agent==tabu_agent:
                continue
            # init
            time_counter = 0
            current_pos = agent.current_position.copy()
            # first position
            if len(agent.route)>0:
                reservation_table[0][current_pos[0]][current_pos[1]] = agent.id
            time_counter += 1
            # part of route
            for step in agent.route:
                if step=="C" or step=="A" or step=="T":
                    pass
                elif step=="N":
                    current_pos[1] += 1
                elif step=="S":
                    current_pos[1] -= 1
                elif step=="E":
                    current_pos[0] += 1
                elif step=="W":
                    current_pos[0] -= 1
                reservation_table[time_counter][current_pos[0]][current_pos[1]] = agent.id
                time_counter +=1
                if time_counter == reservation_table.shape[0]:
                    break
            # end
            while time_counter < time_horizon:
                reservation_table[time_counter][current_pos[0]][current_pos[1]] = agent.id
                time_counter +=1       
        return reservation_table


###############################################################################
###### MAIN ###################################################################
###############################################################################
environment = Environment(grid_size=GRID_SIZE, route_control=SELECTED_ROUTE_CONTROL)
# spawn initial agents
for n in range(0, 10):#int(N_AGENTS/2)):
    environment.spawn_agent()
# simulation loop
SIMULATION_TIME_STEPS = 100
SIMULATION_TIME_STEPS_STOP_SPAWNING = 70
while environment.time < SIMULATION_TIME_STEPS:
    print("\n\ntime:", environment.time, "\t| agents:", len(environment.agents), "\t| tasks:", len(environment.tasks))
    # general update
    environment.time += 1
    # handle agents
    environment.handle_agents()
    # # spawn tasks randomly
    if environment.time < SIMULATION_TIME_STEPS_STOP_SPAWNING:
        if len(environment.tasks)<len(environment.agents):
            environment.spawn_task()
        else:
            if np.random.random()>0.9:
                if len(environment.tasks)*2+len(environment.agents)<100-30:
                    environment.spawn_task()
    # handle tasks
    environment.assign_open_tasks()
    closed = environment.close_finished_tasks()
    # visualize
    plot_environment_and_reservation(environment, time_horizon=VISUALIZATION_TIME_HORIZON, save_filename=f"figs/x_image_{environment.time:04d}.png")
    # report A-STAR Calls
    print("\tA-Star Calls:", AStarPathPlanner.COUNTER)
    AStarPathPlanner.COUNTER = 0
    
make_gif(input_pattern="figs/x_image_*.png", output_gif=f"animation_{SELECTED_ROUTE_CONTROL}.gif", duration=0.2)
