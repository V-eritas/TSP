# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from queue import Queue
from copy import deepcopy as dc
# %%


class BAB:
    class Status:
        def __init__(self, num, s, e, lb=0, cost=0, path=[], visited=[]):
            self.s = s
            self.e = e
            self.num = num    # 已经过的城市数量
            self.lb = lb
            self.cost = cost
            self.path = dc(path)
            self.visited = dc(visited)
        # 代价小的优先级高
        # def __cmp__(self, another):
        #     return self.cost < another.cost

    def __init__(self, points):
        self.n = points.shape[0]
        self.p = points
        self.dist = cdist(self.p, self.p, metric="Euclidean")
        self.path = []
        self.res = np.inf
        for i in range(self.n):
            self.dist[i][i] = np.inf
        self.__init_lower_bound()
        self.__init_upper_bound()

    def __init_upper_bound(self):
        visited = np.zeros(self.n)
        visited[0] = 1
        Status, mincost = 0, 0
        for i in range(self.n - 1):
            new_cost = np.inf
            for j in range(1, self.n):
                if not visited[j] and new_cost > self.dist[Status, j]:
                    new_cost = self.dist[Status][j]
                    new_Status = j
            mincost += new_cost
            visited[new_Status] = 1
            Status = new_Status
        assert visited.sum() == self.n
        mincost += self.dist[Status][0]
        self.upper_bound = mincost
        del visited
        return self.upper_bound

    def __init_lower_bound(self):
        self.lower_bound = 0
        min_index = np.argpartition(self.dist, 2, axis=1)
        for i, ind in enumerate(min_index):
            self.lower_bound += self.dist[i, ind[:2]].sum()
        self.lower_bound = self.lower_bound/2.0
        return self.lower_bound

    def __get_lower_bound(self, cur_status):
        lb = cur_status.cost*2
        min1 = np.inf
        min2 = np.inf

        # 起点到最近未遍历城市的距离
        for i in range(self.n):
            if cur_status.visited[i] == False and min1 > self.dist[i][cur_status.s]:
                min1 = self.dist[i][cur_status.s]
        # 终点到最近未遍历城市的距离
        for i in range(self.n):
            if cur_status.visited[i] == 0 and min2 > self.dist[cur_status.e][i]:
                min2 = self.dist[cur_status.e][i]
        lb = lb+min1+min2

        # 进入并离开每个未遍历城市的最小成本
        for i in range(self.n):
            if cur_status.visited[i]:
                continue
            min_2 = self.dist[i, np.argpartition(self.dist[i], 2)[:2]].sum()
            lb += min_2
        return lb/2

    def solve(self):
        res = np.inf
        Sq = Queue()
        root = self.Status(num=1, s=0, e=0,
                           lb=self.lower_bound, path=[0], visited=np.zeros(self.n))
        root.visited[0] = 1
        Sq.put(root)
        while not Sq.empty():
            cur_status = Sq.get()
            # 访问到最后一个城市时更新全局上界，若已达到全局下界则退出
            if cur_status.num == self.n-1:
                p = np.argmin(cur_status.visited)
                total_cost = cur_status.cost + \
                    self.dist[p][cur_status.s]+self.dist[cur_status.e][p]
                res = min(total_cost, res)
                if total_cost <= cur_status.lb:
                    self.res = res
                    self.path = dc(cur_status.path)
                    self.path.extend([p, 0])
                    break
                else:
                    self.upper_bound = min(total_cost, self.upper_bound)
                    continue
            # 不是最后一个城市
            for i in range(self.n):
                if cur_status.visited[i] == 0:
                    next_status = self.Status(num=cur_status.num+1, s=cur_status.s, e=i, cost=cur_status.cost +
                                              self.dist[cur_status.e][i], path=cur_status.path, visited=cur_status.visited)
                    next_status.visited[i] = 1
                    next_status.path.append(i)
                    next_status.lb = self.__get_lower_bound(next_status)
                    if next_status.lb < self.upper_bound:
                        Sq.put(next_status)
        # self.show()
        return self.path, self.res

    def show(self, figname='BAB_10_cities.png', annotation_size=8, dpi=200, save=True):
        plt.figure(figsize=(5,5))
        plt.plot(self.p[self.path, 0],self.p[self.path, 1], c='r', label="BAB")
        plt.scatter(self.p[self.path, 0],self.p[self.path, 1], s=4)
        plt.legend(loc='best')
        for i in self.path:
            plt.annotate(i, xy=self.p[i], xytext=self.p[i]+(4, 4), size=annotation_size)
        if save:
            plt.savefig('figure/'+figname, dpi=200)
        plt.show()
        plt.gcf().clear()


# %%
if __name__ == '__main__':
    p = pd.read_csv(r'data\TSP10cities.tsp',
                    sep=' ', header=None, index_col=0).values

    times = np.zeros(10)
    i = 0
    while i < len(times):
        a = BAB(p)
        start = time.clock()
        path, mincost = a.solve()
        end = time.clock()
        times[i] = end-start
        print("{}/{} cost time: {} ".format(i, len(times), end-start))
        i = i+1
    a.show()
    print('best self.path is: {0}, with total_cost = {1:6f}'.format(
        ' -> '.join(str(i) for i in path), mincost))
    print('The average running time is {0:.4f} +/- {1:.4f} us '.format(
        1e6*times.mean(), 1e6*times.std()))

# %%
