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
    def _karma_payment_rule(
        cost_mine: int, cost_other: int, other_resolves_conflict: bool, karma_params
    ) -> int:
        # payment = karma_params["karma_payment"]
        # payment = max(0, cost_mine - cost_other) if other_resolves_conflict else max(0, cost_other - cost_mine)
        payment = cost_other if other_resolves_conflict else cost_mine
        return payment

    @staticmethod
    def negotiate_karma(
        cost_other: int,
        cost_mine: int,
        agent_other: Agent,
        agent_self: Agent,
        karma_params,
    ) -> bool:
        # agreement_to_solve_conflict: bool
        # if cost_other <= 0:
        #     agreement_to_solve_conflict = True
        # else:
        #     if cost_mine > cost_other:
        #         cost_delta = cost_mine - cost_other
        #         if cost_delta >= karma_params["delta_threshold"]:
        #             if agent_self.karma_balance >= karma_params["karma_payment"]:
        #                 agreement_to_solve_conflict = True
        #                 agent_self.karma_balance = (
        #                     agent_self.karma_balance - karma_params["karma_payment"]
        #                 )
        #                 agent_other.karma_balance = (
        #                     agent_other.karma_balance + karma_params["karma_payment"]
        #                 )
        #             else:
        #                 agreement_to_solve_conflict = False
        #         else:
        #             agreement_to_solve_conflict = False
        #     else:
        #         agreement_to_solve_conflict = False
        # return agreement_to_solve_conflict

        # initialize negotiation agreement parameters
        other_resolves_conflict: bool

        # compute cost values considering past behavior (karma balance)
        cost_other_adjusted = (
            cost_other + karma_params["karma_influence"] * agent_other.karma_balance
        )
        cost_mine_adjusted = (
            cost_mine + karma_params["karma_influence"] * agent_self.karma_balance
        )

        if cost_mine_adjusted - cost_other_adjusted > karma_params["delta_threshold"]:
            # if the agent's own cost is sufficiently higher than the other agent's cost, the other agent has to resolve the conflict
            other_resolves_conflict = True
        elif np.isclose(
            cost_mine_adjusted - cost_other_adjusted,
            karma_params["delta_threshold"],
            atol=1e-3,
        ):
            # if the difference of adjusted costs is close to the threshold, we randomize the decision to avoid systematic bias
            other_resolves_conflict = np.random.choice([True, False])
        else:
            # if difference of augmented costs is zero or below threshold, this agent has to resolve the conflict
            other_resolves_conflict = False

        payment = NegotiationStrategy._karma_payment_rule(
            cost_mine,
            cost_other,
            other_resolves_conflict=other_resolves_conflict,
            karma_params=karma_params,
        )
        if other_resolves_conflict:
            agent_self.karma_balance -= payment
            agent_other.karma_balance += payment
        else:
            agent_self.karma_balance += payment
            agent_other.karma_balance -= payment

        return other_resolves_conflict
