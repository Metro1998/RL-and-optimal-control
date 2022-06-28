import math
import numpy as np


class DynamicProgramming:
    """
    An analytical method for RL problem with priori known information.
    """

    def __init__(self, T, initial_state):
        """

        :param T: the time steps in one episode.
        """

        self.initial_state = initial_state
        self.T = T
        self.t = [0]
        self.t_counter = T - 1

    def backward(self, a, b, c):
        """
        Calculate the t sequence, more details about a please refer to readme 3.1.1

        :param a: a * x**2 + b * x + c
        :param b:
        :param c:
        :return:
        """
        t = (- b ** 2 + 4 * a * c) / (4 * a)
        self.t.append(t)
        self.t_counter -= 1
        if self.t_counter > 0:
            return self.backward(a=1 + t, b=2 * t, c=1 + t)
        else:
            pass

    @staticmethod
    def step(x, u):
        """
        Environment steps
        :param x:
        :param u:
        :return:
        """
        return x + u

    def cost_over_all(self, optimal_initial_action):
        """
        Calculate the cost over all.

        :return:
        """

        return (1 + self.t[-1]) * (self.initial_state ** 2 + optimal_initial_action ** 2) + \
               2 * self.t[-1] * self.initial_state * optimal_initial_action


if __name__ == "__main__":
    #Dynamic Programming
    dp = DynamicProgramming(T=10, initial_state=4)
    dp.backward(a=1, b=0, c=1)

    optimal_actions = []
    x = dp.initial_state
    for i in dp.t[::-1]:
        optimal_action = - i / (1 + i) * x
        x = dp.step(x, optimal_action)
        optimal_actions.append(optimal_action)

    cost = dp.cost_over_all(optimal_initial_action=optimal_actions[0])
    print('Under exact dynamic programming, the cumulative cost is {}'.format(cost))

    # 3.1.2 Value Iteration
    initial_state = 4
    x = initial_state  # intialize state
    k_star = (1 + math.sqrt(5)) / 2
    optimal_actions_ = []
    for i in range(10):
        optimal_action_ = - (k_star * x) / (1 + k_star)
        optimal_actions_.append(optimal_action_)

        dp.step(x, optimal_action_)

    cost = k_star * initial_state ** 2
    print('Under value iteration, the cumulative cost is {}'.format(cost))
