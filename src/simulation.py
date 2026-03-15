"""
POTENTIAL TITLE: 
    KARMA MECHANISMS FOR DECENTRALIZED, ORIENTATION-AWARE MAPF
    
interesting repo: https://github.com/GavinPHR/Multi-Agent-Path-Finding?tab=readme-ov-file
"""

###############################################################################
###### IMPORTS ################################################################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment


###############################################################################
###### CONSTANTS ##############################################################
###############################################################################

AGENT_STATUS_IDLE = 0
AGENT_STATUS_PICKUP = 1
AGENT_STATUS_CARRY = 2
AGENT_STATUS_DROPOFF = 3

AGENT_ORIENTATION_NORTH = 0
AGENT_ORIENTATION_EAST = 1
AGENT_ORIENTATION_SOUTH = 2
AGENT_ORIENTATION_WEST = 3
AGENT_ORIENTATIONS = [AGENT_ORIENTATION_NORTH, AGENT_ORIENTATION_EAST, AGENT_ORIENTATION_SOUTH, AGENT_ORIENTATION_WEST]

SQUARE_SYMBOL_EMPTY = 0
SQUARE_SYMBOL_OCCUPIED = 1




###############################################################################
###### PARAMETERS #############################################################
###############################################################################

GRID_SIZE = 10
N_AGENTS = 10
TIME_HORIZON = 100



###############################################################################
###### CLASSES ################################################################
###############################################################################

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
        """
        Estimate travel time including rotation cost.
        """
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

class ReservationTable:
    def __init__(self):
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
        # end
        while time_counter < TIME_HORIZON-1:
            self.reservation_table[time_counter][current_pos[0]][current_pos[1]] = agent.id
            time_counter +=1            
        
    def unreserve(self, agent):
        self.reservation_table[self.reservation_table == agent.id] = 0
    
    def is_reserved(self, t, x, y):
        return self.reservation_table[t][x][y]==0
    

    
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
        self.tasks.append(
            Task(
                task_id=self.determine_new_id(self.tasks), 
                from_position=self.grid.get_random_empty_square(),
                to_position=self.grid.get_random_empty_square(),
                grid=self.grid,
                time=self.time
            )
        )
        
    def find_closest_idle_agent(self, position):
         lst_idle_agents = [agent for agent in self.agents if agent.is_idle()]
         if not lst_idle_agents:
             return None
         # find closest agent to task.from_position
         best_agent = None
         best_dist = None
         for agent in lst_idle_agents:
             # Manhattan distance; use Euclidean if you prefer
             dist = Geometry.travel_time_with_rotation(position, agent.current_position, agent.current_orientation)
             if best_dist is None or dist < best_dist:
                 best_dist = dist
                 best_agent = agent
         return best_agent 
    
    """ This implementation is just based on order of tasks
    def assign_open_tasks(self):
        lst_unassigned_tasks = [task for task in self.tasks if not task.is_assigned()]
        for task in lst_unassigned_tasks:
            best_agent = self.find_closest_idle_agent(position=task.from_position)
            if best_agent is not None:
                best_agent.assigned_task = task.id
                best_agent.target_position = task.from_position
                task.assigned_agent = best_agent.id
    """
    
    """ This implementation considers distances when assigning for better total time.
    def assign_open_tasks(self): 
        #
        idle_agents = [a for a in self.agents if a.is_idle()]
        open_tasks = [t for t in self.tasks if not t.is_assigned()]
        if not idle_agents or not open_tasks:
            return
        alpha = 0.2
        candidates = []
        for agent in idle_agents:
            for task in open_tasks:
                dist = Geometry.travel_time_with_rotation(
                    agent.current_position,
                    task.from_position,
                    agent.current_orientation
                )
                wait_time = self.time - task.spawned_time
                cost = dist - alpha * wait_time
                candidates.append((cost, agent, task))
        # best candidates first
        candidates.sort(key=lambda x: x[0])
        used_agents = set()
        used_tasks = set()
        for cost, agent, task in candidates:
            if agent.id in used_agents:
                continue
            if task.id in used_tasks:
                continue
            agent.assigned_task = task.id
            agent.target_position = task.from_position
            task.assigned_agent = agent.id
            used_agents.add(agent.id)
            used_tasks.add(task.id)
    """
    
    def assign_open_tasks(self, alpha=0.2):
        """This implementation furthermore assigns tasks optimally using Hungarian algorithm."""
        idle_agents = [a for a in self.agents if a.is_idle()]
        open_tasks = [t for t in self.tasks if not t.is_assigned()]
        if not idle_agents or not open_tasks:
            return
        num_agents = len(idle_agents)
        num_tasks = len(open_tasks)
        # cost matrix
        cost_matrix = np.zeros((num_agents, num_tasks))
        for i, agent in enumerate(idle_agents):
            for j, task in enumerate(open_tasks):
                dist = Geometry.travel_time_with_rotation(
                    agent.current_position,
                    task.from_position,
                    agent.current_orientation
                )
                wait_time = self.time - task.spawned_time
                cost = dist - alpha * wait_time    
                cost_matrix[i, j] = cost
        # Hungarian algorithm
        agent_indices, task_indices = linear_sum_assignment(cost_matrix)
        # apply assignments
        for a_idx, t_idx in zip(agent_indices, task_indices):
            agent = idle_agents[a_idx]
            task = open_tasks[t_idx]
            agent.assigned_task = task.id
            agent.target_position = task.from_position
            task.assigned_agent = agent.id
            
    def close_finished_tasks(self):
        finished_tasks = [task for task in self.tasks if task.is_finished()]
        for task in finished_tasks:
            for agent in self.agents:
                if agent.id == task.assigned_agent:
                    agent.release_task()
            self.tasks.remove(task)
            task.completed_time = self.time
            self.completed_tasks.append(task)


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

    def occupy(self, position):
        self.occupancy_grid[position[0], position[1]] = SQUARE_SYMBOL_OCCUPIED

    def release(self, position):
        self.occupancy_grid[position[0], position[1]] = SQUARE_SYMBOL_EMPTY



class Task:
    def __init__(self, task_id, from_position, to_position, grid, time):
        self.id = task_id
        self.current_position = from_position.copy()
        self.from_position = from_position.copy()
        self.to_position = to_position.copy()
        self.assigned_agent = None
        self.grid = grid
        self.grid.occupy(self.current_position)
        self.spawned_time = time
        self.completed_time = None
        
    def is_assigned(self):
        return self.assigned_agent is not None

    def is_finished(self):
        return self.current_position==self.to_position



class Agent:
    def __init__(self, agent_id, grid):
        self.id = agent_id
        self.grid = grid
        self.current_position = self.grid.get_random_empty_square()
        self.current_orientation = np.random.choice(AGENT_ORIENTATIONS)
        self.assigned_task = None
        self.status = AGENT_STATUS_IDLE
        self.route = ""
        self.target_position = []
        self.grid.occupy(self.current_position)
        
    def is_idle(self):
        return self.assigned_task is None
    
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
            self.target_position = self.assigned_task
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
    


###############################################################################
###### METHODS ################################################################
###############################################################################
def plot_grid(environment, save_filename=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Hide axes
    ax.set_xlim(-0.5, environment.grid.grid_size - 0.5)
    ax.set_ylim(-0.5, environment.grid.grid_size - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw light gray grid squares (slightly smaller than full size)
    grid_size = environment.grid.grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            # Color based on occupancy
            is_occupied = environment.grid.occupancy_grid[i, j] == SQUARE_SYMBOL_OCCUPIED
            facecolor = '#d0d0d0' if is_occupied else '#f0f0f0'  # darker gray vs light gray
            # 0.85 size creates white lines between squares
            rect = patches.Rectangle((i-0.425, j-0.425), 0.85, 0.85, 
                                   linewidth=0.5, edgecolor='white', 
                                   facecolor=facecolor)
            ax.add_patch(rect)
            ax.text(i, j-0.5, f'({i},{j})', ha='center', va='center', fontsize=6, fontweight='bold', color='darkgray')
             
    # Plot CONNECTIONS between agents and their assigned tasks (blue dashed lines)
    task_size = 0.2
    for agent in environment.agents:
        if agent.assigned_task is not None:  # Agent has a task assigned
            # Find the task by ID
            assigned_task = next((task for task in environment.tasks if task.id == agent.assigned_task), None)
            if assigned_task:
                # # Draw red dashed line from agent to task.from_position
                # ax.plot([agent.current_position[0], assigned_task.from_position[0]],
                #        [agent.current_position[1], assigned_task.from_position[1]],
                #        'r--', linewidth=2, alpha=0.3)
                # # Draw blue dashed line from agent to task.from_position
                # ax.plot([assigned_task.from_position[0], assigned_task.to_position[0]],
                #        [assigned_task.from_position[1], assigned_task.to_position[1]],
                #        'b--', linewidth=2, alpha=0.3)
                
                # Draw red arrow from agent to task.from_position
                ax.annotate(
                    '', 
                    xy=assigned_task.from_position, xytext=agent.current_position,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.5)
                )
                # Draw blue arrow from task.from_position to task.to_position
                ax.annotate(
                    '', 
                    xy=assigned_task.to_position, xytext=assigned_task.from_position,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.5)
                )
                
                # Draw blue rectangle at target position (to_position) with alpha=0.3
                target_rect = patches.Rectangle((
                    assigned_task.to_position[0]-task_size/2, 
                    assigned_task.to_position[1]-task_size/2), 
                    task_size, task_size, 
                    linewidth=2, 
                    edgecolor='blue', 
                    facecolor='blue', 
                    alpha=0.3)
                ax.add_patch(target_rect)
                
    # Plot tasks (small filled black squares)
    for task in environment.tasks:
        rect = patches.Rectangle((
            task.current_position[0]-task_size/2, 
            task.current_position[1]-task_size/2), 
            task_size, 
            task_size, 
            linewidth=1, 
            edgecolor='black', 
            facecolor='black')
        ax.add_patch(rect)
    
    # Plot agents (circles with orientation lines)
    for agent in environment.agents:
        # Circle color based on status
        color_map = {
            AGENT_STATUS_IDLE: 'darkgray',
            AGENT_STATUS_PICKUP: 'red', 
            AGENT_STATUS_CARRY: 'orange',
            AGENT_STATUS_DROPOFF: 'green'
        }
        circle_color = color_map.get(agent.status, 'darkgray')
        
        # Agent circle (unfilled)
        circle = patches.Circle(agent.current_position, 0.3, 
                              linewidth=2, edgecolor=circle_color, facecolor='none')
        ax.add_patch(circle)
        
        # Orientation line (only for non-idle agents)
        center_x, center_y = agent.current_position
        length = 0.3
        if agent.current_orientation == AGENT_ORIENTATION_NORTH:
            dx, dy = 0, length
        elif agent.current_orientation == AGENT_ORIENTATION_EAST:
            dx, dy = length, 0
        elif agent.current_orientation == AGENT_ORIENTATION_SOUTH:
            dx, dy = 0, -length
        else:  # WEST
            dx, dy = -length, 0
            
        line = Line2D([center_x + dx/2, center_x + dx], [center_y + dy/2, center_y + dy],
                     color=circle_color, linewidth=3)
        ax.add_line(line)
    
    plt.title(f't = {environment.time}', fontsize=16, pad=20)
    
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight', facecolor='white')
    else:
        plt.show()

    
def draw_reservation_table(reservation_table):
    import plotly.io as pio
    pio.renderers.default = "browser"   # opens in your default web browser
    import plotly.graph_objects as go
    rt = reservation_table.reservation_table  # shape: (T, X, Y)
    # indices of reserved cells
    t_idx, x_idx, y_idx = np.where(rt > 0)
    agent_ids = rt[t_idx, x_idx, y_idx]
    fig = go.Figure(data=go.Scatter3d(
        x=x_idx,          # X axis
        y=y_idx,          # Y axis
        z=t_idx,          # time axis
        mode='markers',
        marker=dict(
            size=4,
            color=agent_ids,    # color by agent id
            colorscale='Viridis',
            opacity=0.3,
            showscale=True
        )
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Time',
            xaxis=dict(nticks=GRID_SIZE+1, range=[-0.5, GRID_SIZE-0.5]),
            yaxis=dict(nticks=GRID_SIZE+1, range=[-0.5, GRID_SIZE-0.5]),
            zaxis=dict(nticks=6, range=[-0.5, TIME_HORIZON-0.5]),
            aspectmode='cube'
        )
    )
    fig.show()



"""
###############################################################################
###### MAIN ###################################################################
###############################################################################

environment = Environment(grid_size=GRID_SIZE)
# spawn agents
for n in range(0, N_AGENTS):
    environment.spawn_agent()
# spawn tasks
for n in range(0, int(N_AGENTS/2)):
    environment.spawn_task()
# assign tasks
environment.assign_open_tasks()

plot_grid(environment)


# PLAN
reservation_table = ReservationTable()
for agent in environment.agents:
    reservation_table.reserve_agent_route(agent)

draw_reservation_table(reservation_table)
"""



environment = Environment(grid_size=GRID_SIZE)
# spawn agents
environment.spawn_agent()
# spawn tasks
for n in range(0, 3):#int(N_AGENTS/2)):
    environment.spawn_task()

# plot_grid(environment)

# assign tasks
environment.assign_open_tasks()
plot_grid(environment)

# plan route
     
from planner_central_CBS import Planner_CBS

grid = [
    [0,0,0,0],
    [0,1,0,0],
    [0,0,0,0]
]

starts = [
    (0,0,1),  # x,y,theta
    (3,2,3)
]

goals = [
    (3,2),
    (0,2)
]


grid = environment.grid.occupancy_grid
starts = [(agent.current_position[0], agent.current_position[1], agent.current_orientation) for agent in environment.agents]
goals = [(agent.target_position[0], agent.target_position[1]) for agent in environment.agents if not agent.is_idle()]

planner = Planner_CBS()
routes = planner.plan(grid,starts,goals)

print(routes)



# for i,p in enumerate(paths):
#     print("agent",i)
#     for s in p:
#         print(s)

# routes = convert_paths_to_routes(paths)

# print(routes)