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
        agent_other: "Agent",
        agent_self: "Agent",
        lambda_: float = 0.1,
        gamma: int = 1,
    ) -> bool:
        # effective cost = immediate cost - weighted karma
        score_other: float = cost_other - lambda_ * agent_other.karma_balance
        score_self: float = cost_mine - lambda_ * agent_self.karma_balance

        # decide who replans
        if score_other < score_self:
            # other agent replans → reward cooperation
            agent_other.karma_balance += gamma
            agent_self.karma_balance -= gamma
            return True
        else:
            # self replans → reward cooperation
            agent_other.karma_balance -= gamma
            agent_self.karma_balance += gamma
            return False
