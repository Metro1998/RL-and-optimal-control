import numpy as np
import copy
from scipy.optimize import minimize
from utils import visualize


class ModelPredictiveController:
    """
    The implementation of model predictive controller (MPC).
    """

    def __init__(self, N, initial_x, x_constrained, u_constrained):
        """

        :param N:
        :param initial_x:
        :param x_constrained:
        :param u_constrained:
        """
        self.transition_matrix_0 = np.array([
            [1, 1],
            [0, 1]])

        self.transition_matrix_1 = np.array([
            [0],
            [1]
        ])

        self.N = N
        self.initial_x = initial_x
        self.x_constrained = x_constrained
        self.u_constrained = u_constrained

    def set_initial_state(self, x):
        self.initial_x = x

    def step(self, x, u):
        """
        Environment steps according.
        :param x: the last state
        :param u: the executed action
        :return:
        """

        return np.dot(self.transition_matrix_0, x) + np.dot(self.transition_matrix_1, u)

    @staticmethod
    def cost_one_step(x, u):
        """
        Calculate the cost for one step.
        :param x:
        :param u:
        :return:
        """

        return 1 / 2 * (x[0] ** 2 + x[1] ** 2) + 1 / 2 * 10 * u ** 2

    def cost_N_steps(self, u):
        """
        Calculate the cost for N steps.
        :param u: List of candidate actions.
        :return:
        """
        cost = 0
        x = copy.deepcopy(self.initial_x)
        for i in range(self.N):
            cost += self.cost_one_step(x, u[i])
            x = self.step(x, u[i])

        # Plus the state value of x_k+N
        cost += 1 / 2 * (x[0] ** 2 + x[1] ** 2)

        return cost

    def constrained_controllability_condition_0(self, u):
        """
        Calculate the constraints of controllability condition.
        :param u:
        :return:
        """
        x = copy.deepcopy(self.initial_x)
        for i in range(self.N):
            x = self.step(x, u[i])

        # x is a func of u (sequence of action)
        return x[0]

    def constrained_controllability_condition_1(self, u):
        """
        Calculate the constraints of controllability condition.
        :param u:
        :return:
        """
        x = copy.deepcopy(self.initial_x)
        for i in range(self.N):

            x = self.step(x, u[i])

        # x is a func of u (sequence of action)
        return x[1]

    def solve_mpc(self):
        """
        Find the control sequence [u_k, u_k+1, ... u_k+N-1]
        :return:
        """

        cons = [{'type': 'eq', 'fun': self.constrained_controllability_condition_0},
                {'type': 'eq', 'fun': self.constrained_controllability_condition_1},
                # -|u| + u_constrained + e = 0 => u_constrained > |u|
                # {'type': 'ineq', 'fun': lambda x: - abs(x)}
                ]
        b = (-self.u_constrained, self.u_constrained)
        un = (None, None)
        bounds = [b, un, un]

        u0 = np.zeros(self.N)
        res = minimize(self.cost_N_steps, u0, method='SLSQP', bounds=bounds, constraints=cons)

        print('最小值：', res.fun)
        print('最优解：', res.x)
        print('迭代终止是否成功：', res.success)
        print('迭代终止原因：', res.message)

        return res.x


if __name__ == '__main__':
    # Initialize mpc
    mpc = ModelPredictiveController(N=3, initial_x=np.array([[-4.5], [2]]), x_constrained=5, u_constrained=0.5)
    trajectories_overall = []

    # Enter the loop of mpc
    cost = []
    cumulative_cost = 0
    x = mpc.initial_x
    for i in range(11):
        u = mpc.solve_mpc()

        # Simulate the execution of [u_k, u_k+1, ... ,u_k+l-1], and save their result (predictive x including terminal)
        x_predictive = x
        trajectory = [x_predictive.transpose().squeeze()]
        for j in range(len(u)):
            x_predictive = mpc.step(x_predictive, u[j])
            trajectory.append(x_predictive.transpose().squeeze())
        trajectories_overall.append(trajectory)

        cumulative_cost += mpc.cost_one_step(x, u[0])
        # Execute the 1st action and discard the remaining.
        x = mpc.step(x, u[0])
        # record its cost over all
        cost.append(cumulative_cost + 1 / 2 * (x[0] ** 2 + x[1] ** 2))
        # Get the next state and set it as initial_x in the mpc class.
        mpc.set_initial_state(x=x)

    trajectories_overall = np.array(trajectories_overall)
    cost = np.array(cost)
    visualize(trajectories_overall, cost, 'trajectory3.jpg')

