import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from constants import SQUARE_SYMBOL_OCCUPIED, AGENT_STATUS_CARRY, AGENT_STATUS_PICKUP, AGENT_STATUS_IDLE, AGENT_STATUS_DROPOFF
from constants import AGENT_ORIENTATION_SOUTH, AGENT_ORIENTATION_NORTH, AGENT_ORIENTATION_EAST


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
            if agent.assigned_task:
                # Draw red arrow from agent to task.from_position
                if agent.status==AGENT_STATUS_PICKUP:
                    ax.annotate(
                        '', 
                        xy=agent.assigned_task.from_position, xytext=agent.current_position,
                        arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.5)
                    )
                # Draw blue arrow from task.from_position to task.to_position
                ax.annotate(
                    '', 
                    xy=agent.assigned_task.to_position, xytext=agent.assigned_task.current_position,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.5)
                )
                
                # Draw blue rectangle at target position (to_position) with alpha=0.3
                target_rect = patches.Rectangle((
                    agent.assigned_task.to_position[0]-task_size/2, 
                    agent.assigned_task.to_position[1]-task_size/2), 
                    task_size, task_size, 
                    linewidth=2, 
                    edgecolor='blue', 
                    facecolor='blue', 
                    alpha=0.3)
                ax.add_patch(target_rect)
                
    # Plot tasks (small filled black squares)
    for task in environment.tasks:
        color = "black"
        if task.assigned_agent:
            if task.assigned_agent.status == AGENT_STATUS_CARRY:
                color="cyan"
        rect = patches.Rectangle((
            task.current_position[0]-task_size/2, 
            task.current_position[1]-task_size/2), 
            task_size, 
            task_size, 
            linewidth=1, 
            edgecolor=color, 
            facecolor=color)
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
    
    plt.title(f't = {environment.time:04d}  |  agents = {len(environment.agents)}  |  tasks = {len(environment.tasks):02d}', fontsize=16, pad=20)
    
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # <- this prevents the window from opening
    else:
        plt.show()

    
def draw_reservation_table(reservation_table, grid_size, time_horizon):
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
            xaxis=dict(nticks=grid_size+1, range=[-0.5, grid_size-0.5]),
            yaxis=dict(nticks=grid_size+1, range=[-0.5, grid_size-0.5]),
            zaxis=dict(nticks=6, range=[-0.5, time_horizon-0.5]),
            aspectmode='cube'
        )
    )
    fig.show()