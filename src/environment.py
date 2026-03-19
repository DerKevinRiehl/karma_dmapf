import numpy as np
from constants import (
    MAPF_CONTROLLER_CENTRALIZED,
    MAPF_CONTROLLER_DECENTRALIZED_RESPECT,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC,
    MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA,
)
from planner_mapf_central_CBS import Planner_CBS
from planner_assignment_central import Planner_Assignment_Central
from geometry import Grid, GridTools
from task import Task
from negotiation_strategy import NegotiationStrategy
from agent import Agent


class Environment:
    def __init__(self, settings):
        self.settings = settings
        self.grid = Grid(grid_size=self.settings["grid_size"])
        self.static_grid = Grid(grid_size=self.settings["grid_size"])
        self.time = 0
        self.agents = []
        self.tasks = []
        self.completed_tasks = []
        # set random seed
        np.random.seed(self.settings["random_seed"])

    def determine_new_id(self, lst):
        last_agent_id = 0
        if len(lst) > 0:
            last_agent_id = lst[-1].id
        return last_agent_id + 1

    def spawn_agent(self):
        self.agents.append(
            Agent(agent_id=self.determine_new_id(self.agents), environment=self)
        )

    def spawn_task(self):
        try:
            self.tasks.append(
                Task(
                    environment=self,
                    task_id=self.determine_new_id(self.tasks),
                    grid=self.grid,
                    time=self.time,
                )
            )
        except:
            # print("\ttoo crowded, cant spawn right now")
            pass

    def assign_open_tasks(self):
        candidate_agents = [
            a for a in self.agents if a.is_idle() or a.is_available_soon()
        ]
        open_tasks = [t for t in self.tasks if not t.is_assigned()]
        if not candidate_agents or not open_tasks:
            return
        agent_indices, task_indices = Planner_Assignment_Central.plan_assignment(
            candidate_agents, open_tasks, self.time
        )
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
        return len(finished_tasks) > 0

    def handle_agents(self):
        # ROUTE EXECTUION: update agent target and status
        self.handle_agents_route_execution()
        # ROUTE PLANNING: for those who need
        if self.settings["mapf_control"] == MAPF_CONTROLLER_CENTRALIZED:
            self.handle_agents_route_planning_centralized()
        elif self.settings["mapf_control"] == MAPF_CONTROLLER_DECENTRALIZED_RESPECT:
            self.handle_agents_route_planning_decentralized_respect()
        elif (
            self.settings["mapf_control"]
            == MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_EGOISTIC
        ):
            self.handle_agents_route_planning_decentralized_negotiate(
                NegotiationStrategy.negotiate_egoistic
            )
        elif (
            self.settings["mapf_control"]
            == MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_ALTRUISTIC
        ):
            self.handle_agents_route_planning_decentralized_negotiate(
                NegotiationStrategy.negotiate_altruistic
            )
        elif (
            self.settings["mapf_control"]
            == MAPF_CONTROLLER_DECENTRALIZED_NEGOTIATE_KARMA
        ):
            self.handle_agents_route_planning_decentralized_negotiate(
                NegotiationStrategy.negotiate_karma, use_agent_params=True
            )

    def handle_agents_route_execution(self):
        for agent in self.agents:
            agent.execute_route()
            if len(agent.route) == 0 and not agent.is_idle():
                agent.update_target_position()

    def print_debug_log(self):
        if self.settings["debug_statements"]:
            print("")
            print("\tOpen Routes:")
            planning_relevant_agents = [
                agent for agent in self.agents if len(agent.route) > 0
            ]
            for agent in planning_relevant_agents:
                print("\t\t", agent.id, agent.route)
            print("")
            print("\tCurrent Agent Positions:")
            for agent in self.agents:
                print(
                    "\t\t",
                    agent.id,
                    "\t",
                    agent.current_position,
                    "\t| ",
                    agent.target_position,
                )
            print("")
            print("")

    def get_agent(self, agent_id):
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def handle_agents_route_planning_centralized(self):
        """
        This works as follows: centralised planning according to CBS, meaning optimization on joint state space.
        """
        self.print_debug_log()
        # conduct centralized planning for running agents with jobs
        planning_relevant_agents = [
            agent for agent in self.agents if not agent.is_idle()
        ]
        if len(planning_relevant_agents) > 0:
            grid = self.grid.occupancy_grid * 0
            starts = [
                (
                    agent.current_position[0],
                    agent.current_position[1],
                    agent.current_orientation,
                )
                for agent in planning_relevant_agents
            ]
            goals = [
                (agent.target_position[0], agent.target_position[1])
                for agent in planning_relevant_agents
            ]
            planner = Planner_CBS(
                grid,
                cbs_params=self.settings["params_cbs"],
                astar_params=self.settings["params_astar"],
            )
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
        planning_relevant_agents = [
            agent
            for agent in self.agents
            if (not agent.is_idle())
            and len(agent.route) == 0
            and len(agent.target_position) == 2
        ]
        for agent in planning_relevant_agents:
            agent.plan_route_decentralized_respectful()

    def handle_agents_route_planning_decentralized_negotiate(
        self, negotiation_function, use_agent_params=False
    ):
        """
        This works as follows: every new agent will plan its route shortest.
        Then checks if it is conflicting with others.
        If not take it.
        If conflicting, negotiate with that specific one...
            - if other agrees (according to negotiation strategy)
                do change for both agents
            - else
                need to replan considering this restriction
        """
        self.print_debug_log()
        # conduct decentralized planning for running agents with jobs who finished their current route
        planning_relevant_agents = [
            agent
            for agent in self.agents
            if (not agent.is_idle())
            and len(agent.route) == 0
            and len(agent.target_position) == 2
        ]
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
                    if self.settings["debug_statements"]:
                        print(
                            f"\tMax negotiation iterations reached for agent {agent.id}, aborting negotiation"
                        )
                    break
                # rebuild full reservation table each iteration so we always check conflicts against the
                # latest routes other agents may have switched to during negotiation
                reservation_table_complete = GridTools.create_3D_reservation_grid(
                    environment=self,
                    time_horizon=self.settings["params_astar"]["planning_horizon"],
                    agent_list=self.agents,
                    tabu_agent=agent,
                )
                if self.settings["debug_statements"]:
                    print(
                        "\ttrying to finish planning for agent",
                        agent.id,
                        "considered:",
                        len(agents_considered),
                    )
                # determine shortest path (given considered restrictions)
                dynamic_occupancy_grid = GridTools.create_dynamic_occupancy_grid(
                    environment=self,
                    time_horizon=self.settings["params_astar"]["planning_horizon"],
                    agent_list=agents_considered,
                    tabu_agent=agent,
                )
                current_path = agent.path_planner.astar(
                    start=(
                        agent.current_position[0],
                        agent.current_position[1],
                        agent.current_orientation,
                    ),
                    goal=(agent.target_position[0], agent.target_position[1]),
                    dynamic_occupancy=dynamic_occupancy_grid,
                )
                # if cannot plan, just abort for now
                if current_path is None:
                    planning_finished = True
                    break
                # determine conflicts with current plan
                conflicts = GridTools.detect_conflicts(
                    current_path, reservation_table=reservation_table_complete
                )
                # if no conflicts, found the path and can quit
                if len(conflicts) == 0:
                    planning_finished = True
                    break
                # try to solve found conflicts
                if self.settings["debug_statements"]:
                    print(
                        "\tplanning for agent",
                        agent.id,
                        "found",
                        len(conflicts),
                        "conflicts",
                    )
                # try to resolve conflicts with everyone
                all_conflicts_resolved = True
                for conflict in conflicts:
                    if self.settings["debug_statements"]:
                        print("\tchecking the conflict", conflict)
                    conflicting_agent = self.get_agent(conflict["conflicting_agent"])
                    # determine cost other
                    cost_other, alternative_path_other = (
                        conflicting_agent.determine_cost_to_change(
                            to_avoid_path=current_path
                        )
                    )
                    # determine my cost
                    agent.route = agent.path_planner.convert_path_to_route(current_path)
                    cost_mine, alternative_path_mine = agent.determine_cost_to_change(
                        to_avoid_path=conflicting_agent.path_planner.convert_route_to_path(
                            conflicting_agent
                        )
                    )
                    agent.route = []
                    # determine decision - negotiation outcome - egoistic
                    if use_agent_params:
                        agreement_to_solve_conflict = negotiation_function(
                            cost_other, cost_mine, conflicting_agent, agent
                        )
                    else:
                        agreement_to_solve_conflict = negotiation_function(
                            cost_other, cost_mine
                        )

                    # if agrees, continue
                    if agreement_to_solve_conflict:
                        conflicting_agent.change_path_to_satisfy(
                            change_to_path=alternative_path_other
                        )
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
                    else:  # repeating conflicts, avoid inifinite loop
                        current_path = None
                        planning_finished = True
                        break
            # if successful, assign it
            if current_path is not None:
                current_route = agent.path_planner.convert_path_to_route(current_path)
                agent.route = current_route
                if self.settings["debug_statements"]:
                    print("\tsuccessfully done")
            else:
                # otherwise use planning respectfully (conflict-avoiding)
                agent.plan_route_decentralized_respectful()
                if self.settings["debug_statements"]:
                    print("\tnegotiations failed")
