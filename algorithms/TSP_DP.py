import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class Solution:
    def __init__(self, points):
        self.p = points
        self.dist = cdist(self.p, self.p, metric="Euclidean")
        self.n = self.p.shape[0]
        self.dp = np.tile(np.inf, (self.n, 1 << self.n))
        self.route = np.zeros((self.n, 1 << self.n), dtype=np.int32) - 1

    def Search(self):
        for i in range(self.n):
            self.dp[i][0] = self.dist[i][0]
            self.route[i][0] = 0
        self.__DFS(0, (1 << (self.n)) - 2)
        self.__Show1()
        return self.dp[0][(1 << self.n) - 2]

    def __DFS(self, node, unvisited):
        if self.dp[node][unvisited] != np.inf:
            return self.dp[node][unvisited]
        if unvisited == 0:
            return self.dist[node][0]
        mincost = np.inf
        next_node = -1
        for i in range(self.n):
            if 1 & (unvisited >> i):
                new_cost = self.dist[node][i] + self.__DFS(
                    i, unvisited ^ (1 << i))
                if new_cost < mincost:
                    next_node = i
                    mincost = new_cost
        self.dp[node][unvisited] = mincost
        self.route[node][unvisited] = next_node
        self.path = [0]
        sets = (1 << self.n) - 2
        node = 0
        while sets > -1:
            node = self.route[node][sets]
            sets = sets - (1 << node)
            self.path.append(node)
        return self.path[:], mincost

    def __Show1(self):
        print(self.path)
        plt.figure()
        plt.plot(self.p.iloc[self.path, 0],
                 self.p.iloc[self.path, 1], label='DFS')
        plt.show()

    def DP(self):
        self.dp[0][1] = 0
        for i in range(1, self.n):
            self.dp[i][(1 << i) + 1] = self.dist[i][0]
            self.route[i][(1 << i) + 1] = 0
        for visited in range(1, 1 << self.n):
            if not (visited & 1):
                continue
            for i in range(1, self.n):
                if not visited & (1 << i):
                    continue
                for j in range(1, self.n):
                    if not visited & (1 << j) or not i != j:
                        continue
                    if self.dp[i][visited] > self.dp[j][visited - (1 << i)] + self.dist[j][i]:
                        self.dp[i][visited] = self.dp[j][visited -
                                                         (1 << i)] + self.dist[j][i]
                        self.route[i][visited] = j
        mincost = np.inf
        for i in range(1, self.n):
            if mincost > self.dp[i][(1 << self.n) - 1] + self.dist[i][0]:
                mincost = self.dp[i][(1 << self.n) - 1] + self.dist[i][0]
                self.route[0][(1 << self.n) - 1] = i
        self.path = [0]
        sets = (1 << self.n) - 1
        node = self.route[0][sets]
        while sets > 0:
            self.path.append(node)
            node, sets = self.route[node][sets], sets - (1 << node)
        # self.Show2()
        return self.path[:], mincost

    def Show2(self, figname='DP_10_cities.png', annotation_size=8, dpi=200, save=True):
        plt.figure(figsize=(5,5))
        plt.scatter(self.p[self.path, 0], self.p[self.path, 1], marker='*')
        plt.plot(self.p[self.path, 0], self.p[self.path, 1], c='r')
        for i in self.path:
            plt.annotate(i, xy=self.p[i], xytext=self.p[i]+(4, 4), size=annotation_size)
        plt.plot(self.p[self.path, 0], self.p[self.path, 1], c="r", label='DP')
        plt.title('Dynamic Programming for TSP')
        if save:
            plt.savefig('figure/'+figname, dpi=dpi)
        plt.show()
        plt.gcf().clear()


if __name__ == '__main__':
    p = pd.read_csv(r'data\TSP10cities.tsp',
                    sep=' ', header=None, index_col=0).values
    
    times = np.zeros(10)
    i = 0
    while i < len(times):
        a = Solution(p)
        start = time.clock()
        path, mincost = a.DP()
        end = time.clock()
        times[i] = end-start
        print("{}/{} cost time: {} ".format(i, len(times), end-start))
        i = i+1
    a.Show2()
    print('best self.path is: {0}, with total_cost = {1:6f}'.format(
        ' -> '.join(str(i) for i in path), mincost))
    print('The average running time is {0:.4f} +/- {1:.4f} us '.format(
        1e6*times.mean(), 1e6*times.std()))
