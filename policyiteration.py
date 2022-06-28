import numpy as np
from scipy.optimize import minimize


class PolicyIteration:
    """

    """

    def __init__(self, gamma, initial_state, K, H, lamda, mean, std, seed):
        """

        :param gamma:
        :param initial_state:
        :param K: map from state to action, one dimension in the specific problem
        :param H: map from state & action to action value, an array with shape (2, 2)
        :param lamda；the trade off between exploration and exploitation
        :param mean: the mean in normal
        :param std: the std in normal
        :param seed:
        """
        self.gamma = gamma
        self.initial_state = initial_state
        self.K = K
        self.H = H
        self.buffer = []
        self.lamda = lamda
        self.mean = mean
        self.std = std
        self.seed = seed

        # buffer
        self.state = []
        self.action = []
        self.cost = []

    @staticmethod
    def step(x, u):
        """

        :param x: state
        :param u: action
        :return:
        """

        return x + u, x ** 2 + u ** 2

    @staticmethod
    def QFunction(x, u, H):
        """

        :param x:
        :param u:
        :param H: The current Q function mapping
        :return:
        """

        return np.dot(np.array([x, u]), (np.dot(H, np.array([x, u]).transpose())))

    def clear_buffer(self):
        """
        Clear the buffer

        :return:
        """
        self.state = []
        self.action = []
        self.cost = []

    def evaluate(self):
        """
        Evaluate the policy with current trajectories.

        :return:
        """
        H_optimized = self.H

        cons = [{'type': 'eq', 'fun': lambda H_optimized: H_optimized[1] - H_optimized[2]}]
        res = minimize(self.critic_error, H_optimized, method='SLSQP', constraints=cons)

        # print('最小值：', res.fun)
        print('最优解：', res.x)
        # print('迭代终止是否成功：', res.success)
        # print('迭代终止原因：', res.message)

        # update the current Q function mapping
        self.H = res.x.reshape(2, 2)

    def improve(self):
        """
        Improve the policy with current action value.

        :return:
        """

        self.K = - self.H[1][0] / self.H[1][1]

    def critic_error(self, H):
        """

        :param H: The current Q function mapping
        :return:
        """

        H = H.reshape(2, 2)
        Q_error = []
        for i in range(len(self.state) - 1):
            error = (self.QFunction(self.state[i], self.action[i], H) - self.cost[i] -
                     self.gamma * self.QFunction(self.state[i + 1], self.K * self.state[i + 1], H)) ** 2
            Q_error.append(error)

        fro_norm = np.linalg.norm(H)

        return sum(Q_error) + fro_norm

    def sample(self):
        """
        Sample a disturbance from a normal(mean, std ** 2)

        :return:
        """

        return np.random.normal(self.mean, self.std ** 2)


if __name__ == "__main__":
    pi = PolicyIteration(gamma=0.99, initial_state=4, K=1, H=np.array([[1, 1], [1, 1]]), lamda=0.01,
                         mean=0, std=0.1, seed=0)
    episode = 100  # iteration episode
    T = 6  # iteration steps in one episode

    # policy iteration loop
    for i in range(episode):
        pi.clear_buffer()  # clear the buffer
        x = pi.initial_state  # initialize the state
        pi.state.append(x)  # push the initial state into the buffer
        for t in range(T):
            u = pi.K * x + pi.sample()  # mapping from state to action
            pi.action.append(u)  # push the action into the buffer
            x, cost = pi.step(x, u)  # step according the environment
            pi.state.append(x)  # push the state into the buffer
            pi.cost.append(cost)  # push the cost(reward) into the buffer
            pi.evaluate()  # evaluation
        pi.improve()  # improvement

    print('optimal policy mapping: {}'.format(pi.K))
    print('optimal action value mapping: \n {}'.format(pi.H))
    print('Under the optimal policy the trajectories are \n {}'.format(pi.state))
    print('The cumulative cost is {}'.format(sum(pi.cost)))



