
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


