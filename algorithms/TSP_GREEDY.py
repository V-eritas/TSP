import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class TSP:

    def __init__(self, points):
        self.n = points.shape[0]
        self.p = points
        self.dist = cdist(self.p, self.p, metric="Euclidean")
        for i in range(self.n):
            self.dist[i][i] = np.inf
        self.path = np.zeros(self.n + 1, np.int)

    def solve(self):
        node, mincost = 0, 0
        visit = np.zeros(self.n)
        visit[0] = 1
        for i in range(self.n - 1):
            new_cost = np.inf
            for j in range(1, self.n):
                if not visit[j] and new_cost > self.dist[node, j]:
                    new_cost = self.dist[node][j]
                    new_node = j
            mincost += new_cost
            visit[new_node] = 1
            node = new_node
            self.path[i+1] = node
        mincost += self.dist[node][0]
        # print('best path is: {0}, with total_cost = {1}'.format(
            # ' -> '.join(str(i) for i in self.path), mincost))
        # self.show()
        return self.path[:], mincost

    def show(self, figname='GREEDY_100_cities.png', annotation_size=8, dpi=200, save=True):
        plt.figure()
        x, y = self.p[self.path, 0], self.p[self.path, 1]
        plt.plot(x, y, c="r", label=["GREEDY"])
        plt.scatter(x, y, s=4)
        plt.title('Greedy algorithm for TSP')
        plt.savefig('figure/'+figname)
        for i in self.path:
            plt.annotate(
                i, xy=self.p[i], xytext=self.p[i]+(4, 4), size=annotation_size)
        if save:
            plt.savefig('figure/'+figname, dpi=dpi)
        plt.show()


# %%
if __name__ == '__main__':
    p = pd.read_csv(r'data/TSP100cities.tsp',
                    sep=' ', header=None, index_col=0).values
    
    times = np.ones(100)
    i = 0
    while i < 100:
        a = TSP(p)
        start = time.clock()
        path, mincost = a.solve()
        finish = time.clock()
        times[i] = (finish - start)
        i = i + 1
    a.show()
    print('best path is: {0}, with total_cost = {1:6f}'.format(
            ' -> '.join(str(i) for i in path), mincost))
    print('The average running time is {0:.4f} +/- {1:.4f} us '.format(
        1e6*times.mean(), 1e6*times.std()))
