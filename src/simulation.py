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
from constants import ROUTE_CONTROLLER_CENTRALIZED, ROUTE_CONTROLLER_DECENTRALIZED_RESPECT, ROUTE_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC, ROUTE_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC, ROUTE_CONTROLLER_DECENTRALIZED_NEGOTIATE_KAMRA
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
PLANNING_TIME_HORIZON = 50
SIMULATION_TIME_STEPS = 50
INITIAL_KARMA = 5
SELECTED_ROUTE_CONTROL = ROUTE_CONTROLLER_CENTRALIZED
SELECTED_ROUTE_CONTROL = ROUTE_CONTROLLER_DECENTRALIZED_RESPECT
SELECTED_ROUTE_CONTROL = ROUTE_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC
SELECTED_ROUTE_CONTROL = ROUTE_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC

IDLING_NEIGHBORHOOD_SEARCH_RANGE = 2

astar_params = {
    "MAX_STEPS": 5000,
    "MAX_TIME_HORIZON": PLANNING_TIME_HORIZON
}

cbs_params = {
    "MAX_CBS_NODES": 5000,
    "MAX_IDLE_TIME_CONSIDERED": 5,
    "PLANNING_HORIZON": 100
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

    def _determine_intersection_free_path(self, dynamic_occupancy_grid):
        path = self.path_planner.astar(
            start=(self.current_position[0], self.current_position[1], self.current_orientation), 
            goal=(self.target_position[0], self.target_position[1]), 
            dynamic_occupancy=dynamic_occupancy_grid)
        return path 
    
    def _determine_idle_parking_path(self, dynamic_occupancy_grid):
        x0, y0 = self.current_position
        # determine empty cells for idling nearby
        target_candidates = []
        for dx in range(-IDLING_NEIGHBORHOOD_SEARCH_RANGE, IDLING_NEIGHBORHOOD_SEARCH_RANGE+1):
            for dy in range(-IDLING_NEIGHBORHOOD_SEARCH_RANGE, IDLING_NEIGHBORHOOD_SEARCH_RANGE+1):
                # skip the current cell itself
                if dx == 0 and dy == 0:
                    continue
                # explore probe
                probe_pos_x = x0 + dx
                probe_pos_y = y0 + dy
                # grid bounds check
                if probe_pos_x < 0 or probe_pos_y < 0:
                    continue
                if probe_pos_x >= dynamic_occupancy_grid.shape[1] or probe_pos_y >= dynamic_occupancy_grid.shape[2]:
                    continue
                # cell must be free at all times in the horizon
                # dynamic_occupancy_grid[:, x, y] is a 1D boolean array over time
                if not dynamic_occupancy_grid[:, probe_pos_x, probe_pos_y].any():
                    target_candidates.append([probe_pos_x, probe_pos_y])
        # sort them closest to origin (self.current_position)
        target_candidates.sort(
            key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2
        )  # squared distance is enough for ordering[web:19][web:22]
        # if some found, check if there is a path to one
        for target_candidate in target_candidates:
            path = self.path_planner.astar(
                start=(self.current_position[0], self.current_position[1], self.current_orientation), 
                goal=(target_candidate[0], target_candidate[1]), 
                dynamic_occupancy=dynamic_occupancy_grid)
            if not path is None:
                return path
        return None
    
    def plan_route_decentralized_respectful(self):
        # determine dynamic_occupancy_grid given all already planned routes
        dynamic_occupancy_grid = self.environment.create_dynamic_occupancy_grid(time_horizon=PLANNING_TIME_HORIZON, agent_list=self.environment.agents, tabu_agent=self)
        # determine possible, intersection free path
        path = self._determine_intersection_free_path(dynamic_occupancy_grid)
        if path is not None:
            route = self.path_planner.convert_path_to_route(path)
            # print("\trouteadded for ", agent.id, route)
            self.route = route
        
    def determine_cost_to_change(self, to_avoid_path):
        current_cost = len(self.route)
        # determine dynamic_occupancy_grid given all already planned routes
        dynamic_occupancy_grid = self.environment.create_dynamic_occupancy_grid(time_horizon=PLANNING_TIME_HORIZON, agent_list=self.environment.agents, tabu_agent=self)
        # add to_avoid_path to dynamic_occupancy grid
        for state in to_avoid_path:
            dynamic_occupancy_grid[state.t][state.x][state.y] = True
            
        # if you have a target
        if len(self.target_position)>0:  
            # determine possible, intersection free path
            changed_path = self._determine_intersection_free_path(dynamic_occupancy_grid)
            if changed_path is not None:
                changed_route = self.path_planner.convert_path_to_route(changed_path)
                changed_cost = len(changed_route)
                return (changed_cost-current_cost), changed_path
        else:
            # determine if there is any free position nearby to idle parking
            changed_path = self._determine_idle_parking_path(dynamic_occupancy_grid)
            return -1, changed_path
        return 1000, changed_path
        
    def change_path_to_satisfy(self, change_to_path):
        alternative_route = self.path_planner.convert_path_to_route(change_to_path)
        self.route = alternative_route
        
    def do_I_agree_to_change_egoistically(self, requested_conflicting_path):
        cost_to_change, alternative_path = self.determine_cost_to_change(requested_conflicting_path)
        if cost_to_change<=0:
            alternative_route = self.path_planner.convert_path_to_route(alternative_path)
            self.route = alternative_route
            return True
        else:
            return False
            
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

    def determine_new_id(self, lst):
        last_agent_id = 0
        if len(lst)>0:
            last_agent_id = lst[-1].id
        return last_agent_id + 1
    
    def spawn_agent(self):
        self.agents.append(Agent(agent_id=self.determine_new_id(self.agents), environment=self))
        
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
        elif self.route_control==ROUTE_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC:
            self.handle_agents_route_planning_decentralized_negotiate_egoistic()
        elif self.route_control==ROUTE_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC:
            self.handle_agents_route_planning_decentralized_negotiate_altruistic()
    
    def handle_agents_route_execution(self):
        for agent in self.agents:
            agent.execute_route()
            if len(agent.route)==0 and not agent.is_idle():
                agent.update_target_position()
    
    def print_debug_log(self):
        print("")
        print("\tOpen Routes:")
        planning_relevant_agents = [agent for agent in self.agents if len(agent.route)>0]
        for agent in planning_relevant_agents:
            print("\t\t",agent.id, agent.route)
        print("")
        print("\tCurrent Agent Positions:")
        for agent in self.agents:
            print("\t\t",agent.id, "\t", agent.current_position, "\t| ", agent.target_position)
        print("")
        print("")
            
    def handle_agents_route_planning_centralized(self):
        """
        This works as follows: centralised planning according to CBS, meaning optimization on joint state space.
        """
        self.print_debug_log()
        # conduct centralized planning for running agents with jobs
        planning_relevant_agents = [agent for agent in self.agents if not agent.is_idle()]
        if len(planning_relevant_agents)>0:
            grid = self.grid.occupancy_grid*0
            starts = [(agent.current_position[0], agent.current_position[1], agent.current_orientation) for agent in planning_relevant_agents]
            goals = [(agent.target_position[0], agent.target_position[1]) for agent in planning_relevant_agents]
            planner = Planner_CBS(grid, cbs_params=cbs_params, astar_params=astar_params)
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
        self.print_debug_log()        
        # conduct decentralized planning for running agents with jobs who finished their current route
        planning_relevant_agents = [agent for agent in self.agents if (not agent.is_idle()) and len(agent.route)==0 and len(agent.target_position)==2]
        for agent in planning_relevant_agents:
            agent.plan_route_decentralized_respectful()
    
    def get_agent(self, agent_id):
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def detect_conflicts(self, path, reservation_table_complete):
        conflicts = []
        conflicting_agents = []
        for state in path:
            t = state.t
            if t >= reservation_table_complete.shape[0]:
                break
            x = state.x
            y = state.y
            occupying_agent = reservation_table_complete[t][x][y]
            if occupying_agent != 0:
                if occupying_agent not in conflicting_agents:
                    conflicts.append({
                        "time": t,
                        "position": (x, y),
                        "conflicting_agent": int(occupying_agent)
                    })
                    conflicting_agents.append(occupying_agent)
        return conflicts

    def handle_agents_route_planning_decentralized_negotiate_egoistic(self):
        """
        This works as follows: every new agent will plan its route shortest.
        Then checks if it is conflicting with others.
        If not take it.
        If conflicting, negotiate with that specific one...
            - if other agrees (as makes no difference to him)
                do change for both agents
            - else
                need to replan considering this restriction
        """
        self.print_debug_log()      
        # conduct decentralized planning for running agents with jobs who finished their current route
        planning_relevant_agents = [agent for agent in self.agents if (not agent.is_idle()) and len(agent.route)==0 and len(agent.target_position)==2]
        for agent in planning_relevant_agents:
            # try for this agent to plan, given the restrictions it step by step considers
            planning_finished = False
            agents_considered = []
            # determine plan with negotiating with others
            current_path = None
            agents_had_conflict_with = []
            # safety guard to avoid infinite negotiation loops
            max_iterations = max(10, len(self.agents) * 2)
            iter_count = 0
            while not planning_finished:
                iter_count += 1
                if iter_count > max_iterations:
                    print(f"\tMax negotiation iterations reached for agent {agent.id}, aborting negotiation")
                    break
                # rebuild full reservation table each iteration so we always check conflicts against the
                # latest routes other agents may have switched to during negotiation
                reservation_table_complete = self.create_3D_reservation_grid(
                    time_horizon=PLANNING_TIME_HORIZON,
                    agent_list=self.agents,
                    tabu_agent=agent
                )
                print("trying to finish planning for agent", agent.id, "considered:", len(agents_considered))
                # determine shortest path (given considered restrictions)
                dynamic_occupancy_grid = self.create_dynamic_occupancy_grid(time_horizon=PLANNING_TIME_HORIZON, agent_list=agents_considered, tabu_agent=agent)
                current_path = agent.path_planner.astar(
                    start=(agent.current_position[0], agent.current_position[1], agent.current_orientation), 
                    goal=(agent.target_position[0], agent.target_position[1]), 
                    dynamic_occupancy=dynamic_occupancy_grid
                )
                # if cannot plan, just abort for now
                if current_path is None:
                    planning_finished = True
                    break
                # determine conflicts with current plan
                conflicts = self.detect_conflicts(current_path, reservation_table_complete)
                # if no conflicts, found the path and can quit
                if len(conflicts)==0:
                    planning_finished = True
                    break
                # try to solve found conflicts
                print("\tPlanning for agent", agent.id, "found", len(conflicts), "conflicts")
                # try to resolve conflicts with everyone
                all_conflicts_resolved = True                
                for conflict in conflicts:
                    print("\tchecking the conflict", conflict)
                    conflicting_agent = self.get_agent(conflict["conflicting_agent"])
                    # call conflicting agent to replan his stuff
                    # see if he minds to do it differently, maybe same duration
                    agreement_to_solve_conflict = conflicting_agent.do_I_agree_to_change_egoistically(requested_conflicting_path=current_path)
                    # if agrees, continue
                    if agreement_to_solve_conflict:
                        continue
                    # otherwise, break the loop
                    else:
                        all_conflicts_resolved = False
                        break
                # if others changed their plans and agreed, we can keep this
                if all_conflicts_resolved:
                    planning_finished = True
                    break
                # otherwise, didnt work out, so we have to add them into our agents_considered constraints
                for conflict in conflicts:
                    conflicting_agent = self.get_agent(conflict["conflicting_agent"])
                    agents_considered.append(conflicting_agent)
                    agents_considered = list(set(agents_considered))
                    if not conflicting_agent in agents_had_conflict_with:
                        agents_had_conflict_with.append(conflicting_agent)
                    else: # repeating conflicts, avoid inifinite loop
                        current_path = None
                        planning_finished = True
                        break
            # if successful, assign it
            if current_path is not None:
                current_route = agent.path_planner.convert_path_to_route(current_path)
                agent.route = current_route
                print("successfully done")
            else:
                # otherwise use planning respectfully (conflict-avoiding)
                agent.plan_route_decentralized_respectful()
                print("negotiations failed")

    def handle_agents_route_planning_decentralized_negotiate_altruistic(self):
        """
        This works as follows: every new agent will plan its route shortest.
        Then checks if it is conflicting with others.
        If not take it.
        If conflicting, negotiate with that specific one...
            - if other not as worse off
                do change for both agents
            - else
                need to replan considering this restriction
        """
        self.print_debug_log()      
        # conduct decentralized planning for running agents with jobs who finished their current route
        planning_relevant_agents = [agent for agent in self.agents if (not agent.is_idle()) and len(agent.route)==0 and len(agent.target_position)==2]
        for agent in planning_relevant_agents:
            # try for this agent to plan, given the restrictions it step by step considers
            planning_finished = False
            agents_considered = []
            # determine plan with negotiating with others
            current_path = None
            agents_had_conflict_with = []
            # safety guard to avoid infinite negotiation loops
            max_iterations = max(10, len(self.agents) * 2)
            iter_count = 0
            while not planning_finished:
                iter_count += 1
                if iter_count > max_iterations:
                    print(f"\tMax negotiation iterations reached for agent {agent.id}, aborting negotiation")
                    break
                # rebuild full reservation table each iteration so we always check conflicts against the
                # latest routes other agents may have switched to during negotiation
                reservation_table_complete = self.create_3D_reservation_grid(
                    time_horizon=PLANNING_TIME_HORIZON,
                    agent_list=self.agents,
                    tabu_agent=agent
                )
                print("trying to finish planning for agent", agent.id, "considered:", len(agents_considered))
                # determine shortest path (given considered restrictions)
                dynamic_occupancy_grid = self.create_dynamic_occupancy_grid(time_horizon=PLANNING_TIME_HORIZON, agent_list=agents_considered, tabu_agent=agent)
                current_path = agent.path_planner.astar(
                    start=(agent.current_position[0], agent.current_position[1], agent.current_orientation), 
                    goal=(agent.target_position[0], agent.target_position[1]), 
                    dynamic_occupancy=dynamic_occupancy_grid
                )
                # if cannot plan, just abort for now
                if current_path is None:
                    planning_finished = True
                    break
                # determine conflicts with current plan
                conflicts = self.detect_conflicts(current_path, reservation_table_complete)
                # if no conflicts, found the path and can quit
                if len(conflicts)==0:
                    planning_finished = True
                    break
                # try to solve found conflicts
                print("\tPlanning for agent", agent.id, "found", len(conflicts), "conflicts")
                # try to resolve conflicts with everyone
                all_conflicts_resolved = True                
                for conflict in conflicts:
                    print("\tchecking the conflict", conflict)
                    conflicting_agent = self.get_agent(conflict["conflicting_agent"])
                    # call conflicting agent to replan his stuff
                    # see if he minds to do it differently, maybe same duration
                    agreement_to_solve_conflict = conflicting_agent.do_I_agree_to_change_egoistically(requested_conflicting_path=current_path)
                    
                    
                    # call conflicting_agent to see his costs
                    cost_other, alternative_path_other = conflicting_agent.determine_cost_to_change(to_avoid_path=current_path)
                    
                    # determine my cost
                    agent.route = agent.path_planner.convert_path_to_route(current_path)
                    cost_mine, alternative_path_mine = agent.determine_cost_to_change(
                        to_avoid_path=conflicting_agent.path_planner.convert_route_to_path(conflicting_agent)
                    )
                    agent.route = []
                    
                    print("\t my cost:", cost_mine," others cost", cost_other," so I win?", cost_mine > cost_other)
                    
                    # who is worse off?
                    if cost_mine > cost_other:
                        agreement_to_solve_conflict = True
                    elif cost_mine < cost_other:
                        agreement_to_solve_conflict = False
                    else:  # cost_mine == cost_other
                        agreement_to_solve_conflict = np.random.choice([True, False])
                    
                    # continue with other conflicts if won
                    if agreement_to_solve_conflict:
                        conflicting_agent.change_path_to_satisfy(change_to_path=alternative_path_other)
                        continue
                    # otherwise, break the loop
                    else:
                        all_conflicts_resolved = False
                        break
                # if others changed their plans and agreed, we can keep this
                if all_conflicts_resolved:
                    planning_finished = True
                    break
                # otherwise, didnt work out, so we have to add them into our agents_considered constraints
                for conflict in conflicts:
                    conflicting_agent = self.get_agent(conflict["conflicting_agent"])
                    agents_considered.append(conflicting_agent)
                    agents_considered = list(set(agents_considered))
                    if not conflicting_agent in agents_had_conflict_with:
                        agents_had_conflict_with.append(conflicting_agent)
                    else: # repeating conflicts, avoid inifinite loop
                        current_path = None
                        planning_finished = True
                        break
            # if successful, assign it
            if current_path is not None:
                current_route = agent.path_planner.convert_path_to_route(current_path)
                agent.route = current_route
                print("successfully done")
            else:
                # otherwise use planning respectfully (conflict-avoiding)
                agent.plan_route_decentralized_respectful()
                print("negotiations failed")
             
                
             
                
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
