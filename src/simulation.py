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
from visualization import plot_environment_and_reservation
from planner_path_central_CBS import Planner_CBS
from planner_assignment_central import Planner_Assignment_Central
from planner_path_tools import AStarPathPlanner


###############################################################################
###### PARAMETERS #############################################################
###############################################################################

SPAWN_BORDER = 1
GRID_SIZE = 14
N_AGENTS = 10
TIME_HORIZON = 10
SIMULATION_TIME_STEPS = 50



###############################################################################
###### CLASSES ################################################################
###############################################################################

class ReservationTable:
    def __init__(self, TIME_HORIZON, GRID_SIZE):
        self.reservation_table = np.zeros((TIME_HORIZON, GRID_SIZE, GRID_SIZE))

    def reserve_agent_route(self, agent):
        # init
        time_counter = 0
        current_pos = agent.current_position.copy()
        # first position
        self.reservation_table[0][current_pos[0]][current_pos[1]] = agent.id
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
            self.reservation_table[time_counter][current_pos[0]][current_pos[1]] = agent.id
            time_counter +=1
            if time_counter == self.reservation_table.shape[0]:
                break
        # end
        while time_counter < TIME_HORIZON-1:
            self.reservation_table[time_counter][current_pos[0]][current_pos[1]] = agent.id
            time_counter +=1            
        
    def unreserve(self, agent):
        self.reservation_table[self.reservation_table == agent.id] = 0
    
    def is_reserved(self, t, x, y):
        return self.reservation_table[t][x][y]==0
    

class Agent:
    def __init__(self, agent_id, grid):
        self.id = agent_id
        self.grid = grid
        self.current_position = self.grid.get_random_empty_square()
        self.current_orientation = np.random.choice(AGENT_ORIENTATIONS)
        self.assigned_task = None
        self.status = AGENT_STATUS_IDLE
        self.route = []
        self.target_position = []
        self.grid.occupy(self.current_position)
        
    def is_idle(self):
        return self.assigned_task is None
    
    def is_available_soon(self):
        return self.status==AGENT_STATUS_CARRY
    
    def release_task(self):
        self.assigned_task = None
        self.status = AGENT_STATUS_IDLE
        self.target_position = []
        self.route = ""
        
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
    def __init__(self, grid_size):
        self.grid = Grid(grid_size=grid_size)
        self.time = 0
        self.agents = []
        self.tasks = []
        self.completed_tasks = []
        
    def determine_new_id(self, lst):
        last_agent_id = 0
        if len(lst)>0:
            last_agent_id = lst[-1].id
        return last_agent_id + 1
    
    def spawn_agent(self):
        self.agents.append(
            Agent(
                agent_id=self.determine_new_id(self.agents), 
                grid=self.grid
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

    def handle_agents_centralized(self):
        # update agent target and status
        for agent in self.agents:
            agent.execute_route()
            if len(agent.route)==0 and not agent.is_idle():
                agent.update_target_position()
        # conduct centralized planning for running agents
        planning_relevant_agents = [agent for agent in self.agents if not agent.is_idle()]
        if len(planning_relevant_agents)>0:
            grid = self.grid.occupancy_grid*0
            starts = [(agent.current_position[0], agent.current_position[1], agent.current_orientation) for agent in planning_relevant_agents]
            goals = [(agent.target_position[0], agent.target_position[1]) for agent in planning_relevant_agents]
            planner = Planner_CBS(grid)
            # print("\tInput for Planner:", starts, goals, "\n", grid)
            routes = planner.plan(starts, goals)
            # update routes
            if routes is not None:
                for idx, agent in enumerate(planning_relevant_agents):
                    agent.route = routes[idx]


###############################################################################
###### MAIN ###################################################################
###############################################################################
environment = Environment(grid_size=GRID_SIZE)
# spawn initial agents
for n in range(0, 10):#int(N_AGENTS/2)):
    environment.spawn_agent()
# spawn initial tasks
for n in range(0, 5):#int(N_AGENTS/2)):
    environment.spawn_task()
# simulation loop
SIMULATION_TIME_STEPS = 50
while environment.time < SIMULATION_TIME_STEPS:
    print("\n\ntime:", environment.time, "agents:", len(environment.agents), "tasks:", len(environment.tasks))
    # general update
    environment.time += 1
    # handle agents
    environment.handle_agents_centralized()
    # # spawn tasks randomly
    if len(environment.tasks)<len(environment.agents):
        environment.spawn_task()
    else:
        if np.random.random()>0.9:
            if len(environment.tasks)*2+len(environment.agents)<100-30:
                environment.spawn_task()
                # print("\tadded task")
    # handle tasks
    environment.assign_open_tasks()
    closed = environment.close_finished_tasks()
    # visualize
    reservation_table = ReservationTable(TIME_HORIZON, GRID_SIZE)
    for agent in environment.agents:
        reservation_table.unreserve(agent)    
        reservation_table.reserve_agent_route(agent)
    plot_environment_and_reservation(environment,reservation_table, save_filename=f"figs/x_image_{environment.time:04d}.png")
    # report A-STAR Calls
    print("\tA-Star Calls:", AStarPathPlanner.COUNTER)
    AStarPathPlanner.COUNTER = 0