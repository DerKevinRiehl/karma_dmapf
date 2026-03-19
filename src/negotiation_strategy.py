import numpy as np


class NegotiationStrategy:

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
