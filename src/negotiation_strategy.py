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
        agreement_to_solve_conflict: bool
        if cost_other <= 0:
            agreement_to_solve_conflict = True
        else:
            agreement_to_solve_conflict = False
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
        agreement_to_solve_conflict: bool
        if cost_other <= 0:
            agreement_to_solve_conflict = True
        else:
            if cost_mine > cost_other:
                cost_delta = cost_mine - cost_other
                if cost_delta >= karma_params["delta_threshold"]:
                    if (
                        agent_self.karma_balance >= agent_other.karma_balance
                        and agent_self.karma_balance >= karma_params["karma_payment"]
                    ):
                        agreement_to_solve_conflict = True
                        agent_self.karma_balance = (
                            agent_self.karma_balance - karma_params["karma_payment"]
                        )
                        agent_other.karma_balance = (
                            agent_other.karma_balance + karma_params["karma_payment"]
                        )
                    else:
                        agreement_to_solve_conflict = False
                else:
                    agreement_to_solve_conflict = False
            else:
                agreement_to_solve_conflict = False
        return agreement_to_solve_conflict
