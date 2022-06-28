import numpy as np


class DynamicProgramming:
    """
    A dynamic programming calculator for shortest path question.
    """

    def __init__(self):
        # the reward mapping, 1st dimension means along x or y
        # i.e. reward_map[0][0][1] means reward from (0, 1) to (0, 2), which is 9
        # while reward_map[1][1][2] means reward from (1, 2) to (1, 3), which is 6
        self.reward_map = np.array([
            [[11, 9, 7],
             [6, 5, 10],
             [5, 10, 7],
             [10, 6, 5]],
            [[10, 8, 7],
             [7, 9, 6],
             [12, 8, 8],
             [6, 9, 7]]
        ])

    def shortest_path(self, x, y):
        """

        :param x: the location of the start point along the x
        :param y: the location of the start point along the y
        :return:
        """
        # the state value of the end point is 0.
        if x == 0 and y == 0:
            return 0
        # when the start point is along the x axis, just retrospect its source along the x axis.
        elif y == 0:
            return self.shortest_path(x - 1, y) + self.reward_map[0][y][x - 1]
        # when the start point is along the y axis, just retrospect its source along the y axis.
        elif x == 0:
            return self.shortest_path(x, y - 1) + self.reward_map[1][x][y - 1]
        else:
            return min(self.shortest_path(x - 1, y) + self.reward_map[0][y][x - 1],
                       self.shortest_path(x, y - 1) + self.reward_map[1][x][y - 1])


class Rollout_with_One_Step_lookahead_Minimization:
    """
    实际上，rollout与上面dp算法的本质区别在于对state value的处理，dp是不停递归计算出来的精确解，而rollout则是在其rollout policy
    i.e. nearest neighbor 计算出来的近似解。
    """

    def __init__(self):
        self.reward_map = np.array([
            [[11, 9, 7],
             [6, 5, 10],
             [5, 10, 7],
             [10, 6, 5]],
            [[10, 8, 7],
             [7, 9, 6],
             [12, 8, 8],
             [6, 9, 7]]
        ])

    def nearest_neighbor(self, x, y):
        """

        :param x: the location of the start point along the x
        :param y: the location of the start point along the y
        :return:
        """
        if x == 0 and y == 0:
            return 0
        elif y == 0:
            return self.nearest_neighbor(x - 1, y) + self.reward_map[0][y][x - 1]
        elif x == 0:
            return self.nearest_neighbor(x, y - 1) + self.reward_map[1][x][y - 1]
        else:
            minimum_index = np.argmin([self.reward_map[0][y][x - 1], self.reward_map[1][x][y - 1]], axis=0)
            if minimum_index:
                return self.nearest_neighbor(x, y - 1) + self.reward_map[1][x][y - 1]
            else:
                return self.nearest_neighbor(x - 1, y) + self.reward_map[0][y][x - 1]

    def shortest_path(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        print('x, y', x, y)
        if x == 0 and y == 0:
            return 0
            # when the start point is along the x axis, just retrospect its source along the x axis.
        elif y == 0:
            return self.shortest_path(x - 1, y) + self.reward_map[0][y][x - 1]
            # when the start point is along the y axis, just retrospect its source along the y axis.
        elif x == 0:
            return self.shortest_path(x, y - 1) + self.reward_map[1][x][y - 1]
        else:
            minimum_index = np.argmin([self.nearest_neighbor(x - 1, y) + self.reward_map[0][y][x - 1],
                                       self.nearest_neighbor(x, y - 1) + self.reward_map[1][x][y - 1]], axis=0)
            if minimum_index:
                return self.shortest_path(x, y - 1) + + self.reward_map[1][x][y - 1]
            else:
                return self.shortest_path(x - 1, y) + self.reward_map[0][y][x - 1]


if __name__ == "__main__":
    dp = DynamicProgramming()
    rollout = Rollout_with_One_Step_lookahead_Minimization()
    res = dp.shortest_path(3, 3)

    # optimality is a matrix, where every node is a optimal solution.
    optimality = []
    for i in range(4):
        for j in range(4):
            optimality.append(dp.shortest_path(i, j))
    print(optimality)

    # 关于寻找路径，这边没有专门写一个demo，因为有点dirty，大致思想就是把最终的optimality打印出来，相减与reward相比
    # 如果相等表示这条路段可行，最终把所有可行的路段连接起来形成路径

    res_prime = rollout.shortest_path(3, 3)
