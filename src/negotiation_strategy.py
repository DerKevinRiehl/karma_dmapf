import numpy as np


class NegotiationStrategy:
    """
    Decide whether 'other' agent should change
    (agreement_to_solve_conflict = True) means other agent replans
    """

    def negotiate_egoistic(cost_other, cost_mine):
        agreement_to_solve_conflict = cost_other <= 0
        return agreement_to_solve_conflict

    def negotiate_altruistic(cost_other, cost_mine):
        # who is worse off?
        if cost_mine > cost_other:
            agreement_to_solve_conflict = True
        elif cost_mine < cost_other:
            agreement_to_solve_conflict = False
        else:  # cost_mine == cost_other
            agreement_to_solve_conflict = np.random.choice([True, False])
        return agreement_to_solve_conflict

    def negotiate_karma(
        cost_other, cost_mine, agent_other, agent_self, lambda_=0.1, gamma=1.0
    ):
        # effective cost = immediate cost - weighted karma
        score_other = cost_other - lambda_ * agent_other.karma_balance
        score_self = cost_mine - lambda_ * agent_self.karma_balance

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
