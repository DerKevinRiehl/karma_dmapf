from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from agent import Agent


class NegotiationStrategy:
    """
    Decide whether 'other' agent should change
    (agreement_to_solve_conflict = True) means other agent replans
    """

    @staticmethod
    def negotiate_egoistic(cost_other: int, cost_mine: int) -> bool:
        agreement_to_solve_conflict: bool = cost_other <= 0
        return agreement_to_solve_conflict

    @staticmethod
    def negotiate_altruistic(cost_other: int, cost_mine: int) -> bool:
        # who is worse off?
        agreement_to_solve_conflict: bool
        if cost_mine > cost_other:
            agreement_to_solve_conflict = True
        elif cost_mine < cost_other:
            agreement_to_solve_conflict = False
        else:  # cost_mine == cost_other
            agreement_to_solve_conflict = np.random.choice([True, False])
        return agreement_to_solve_conflict

    @staticmethod
    def negotiate_karma(
        cost_other: int,
        cost_mine: int,
        agent_other: Agent,
        agent_self: Agent,
        karma_params,
    ) -> bool:
        # effective cost = immediate cost - weighted karma
        score_other: float = cost_other - karma_params["lambda"] * agent_other.karma_balance
        score_self: float = cost_mine - karma_params["lambda"] * agent_self.karma_balance
    
        if score_self > score_other:
            # other agent replans → reward cooperation
            new_other = agent_other.karma_balance + cost_other * karma_params["gamma"]
            new_self = agent_self.karma_balance - cost_mine * karma_params["gamma"]
    
            # only allow trade if both stay >= 0
            if new_other < 0 or new_self < 0:
                agreement_to_solve_conflict = False
                
            agent_other.karma_balance = new_other
            agent_self.karma_balance = new_self
            agreement_to_solve_conflict = True
            
        elif score_self < score_other:
            agreement_to_solve_conflict = False
        else:  # cost_mine == cost_other
            agreement_to_solve_conflict = np.random.choice([True, False])
        return agreement_to_solve_conflict