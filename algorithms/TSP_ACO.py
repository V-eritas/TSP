# %%
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
import math
import bisect
# %%


class ACO:
    # 记录边的费洛蒙,weigth
    class Edge:
        def __init__(self, s, e, weight, init):
            self.s = s
            self.e = e
            self.weight = weight
            self.pheromone = init

        def __str__(self):
            return "start:{0}, end:{1}, weight:{2}, pheromoone:{3}".format(self.s, self.e, self.weight, self.pheromone)
        __repr__ = __str__

    # 蚂蚁
    class Ant:
        def __init__(self, superclass_obj, alpha, beta):
            # 每个蚁群共享一个地图，即superclass中的edges, nodes, n
            self.alpha = alpha
            self.beta = beta
            self.path = None
            self.distance = np.inf
            self.sc = superclass_obj

        def __select_node(self):
            total_weight = 0
            unvisited = [node for node in range(
                self.sc.n) if node not in self.path]
            for u_node in unvisited:
                total_weight += self.sc.edges[self.path[-1]][u_node].weight
            roulette_wheel = np.zeros(len(unvisited))
            tmp = 0
            for i, u_node in enumerate(unvisited):
                tmp += pow(self.sc.edges[self.path[-1]][u_node].pheromone, self.alpha) * \
                    pow((total_weight /
                         self.sc.edges[self.path[-1]][u_node].weight), self.beta)
                roulette_wheel[i] = tmp
            random_value = random.uniform(0, roulette_wheel[-1])
            return unvisited[bisect.bisect(roulette_wheel, random_value)]

        def find_path(self):
            self.path = [random.randint(0, self.sc.n - 1)]
            while len(self.path) < self.sc.n:
                self.path.append(self.__select_node())

        def get_distance(self):
            self.distance = 0
            for i in range(self.sc.n):
                self.distance += self.sc.edges[self.path[i]
                                               ][self.path[(i + 1) % self.sc.n]].weight
            return self.distance

    def __init__(self, colony_size=10, alpha=1.0, beta=6.0, rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, nodes=None):
        '''
        Args:
            nodes:ndarray
                城市列表，每行为一个城市坐标
            colony_size:int
                蚁群大小
            alpha:float
                控制信息素对路径选择的影响
            beta:float
                控制距离对路径选择的影响
            rho:float
                信息素衰减
        '''
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.colony_size = colony_size
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.steps = steps

        self.nodes = nodes
        self.n = len(nodes)
        self.dist = cdist(nodes, nodes, metric='Euclidean')
        self.edges = [[self.Edge(i, j, self.dist[i][j], initial_pheromone)
                       for j in range(self.n)] for i in range(self.n)]
        self.ants = [self.Ant(self, alpha, beta)
                     for _ in range(self.colony_size)]
        self.best_path = []
        self.best_distance = np.inf
        self.distance_curve = np.zeros(steps)

    def __add_pheromone(self, path, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.n):
            # 双向添加费洛蒙
            self.edges[path[i]][path[(i + 1) % self.n]
                                ].pheromone += weight * pheromone_to_add
            self.edges[path[(i + 1) % self.n]][path[i]
                                               ].pheromone += weight * pheromone_to_add

    def __acs(self):
        for step in range(self.steps):
            for ant in self.ants:
                ant.find_path()
                ant.get_distance()
                self.__add_pheromone(ant.path, ant.distance)
                if ant.distance < self.best_distance:
                    self.best_path = ant.path
                    self.best_distance = ant.distance
            self.distance_curve[step] = self.best_distance
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

    def run(self):
        self.__acs()
        # print('Path:{0}->{1}'.format(' -> '.join(str(i) for i in self.best_path), self.best_path[0]))
        # print('Distance:{}\n'.format(self.best_distance))
        return self.best_path, self.best_distance

    def show(self, filename=None, annotation_size=8, dpi=200):
        x = self.nodes[self.best_path, 0].tolist()
        x.append(x[0])
        y = self.nodes[self.best_path, 1].tolist()
        y.append(y[0])
        plt.plot(x, y, c='r', label="path")
        plt.scatter(x, y, s=4)
        for i in self.best_path:
            plt.annotate(
                i, xy=self.nodes[i], xytext=self.nodes[i]+(4, 4), size=annotation_size)
        if filename:
            plt.savefig(filename, dpi=dpi)
        plt.show()
        plt.gcf().clear()

    def plot(self, filename=None, dpi=200):
        plt.figure(figsize=(5,5))
        plt.plot(range(self.steps), self.distance_curve, c='r')
        if filename:
            plt.savefig(filename, dpi=dpi)
        plt.show()
        plt.gcf().clear()


# %%
if __name__ == '__main__':
    _nodes = pd.read_csv(r'data/TSP10cities.tsp',
                         sep=' ', header=None, index_col=0).values
    times = np.ones(1)
    i = 0
    while i < len(times):
        acs = ACO(colony_size=10, steps=50, nodes=_nodes)
        start = time.clock()
        path, mincost = acs.run()
        finish = time.clock()
        times[i] = (finish - start)
        i = i + 1
    print('best path is: {0}, with total_cost = {1:6f}'.format(
        ' -> '.join(str(i) for i in path), mincost))
    print('The average running time is {0:.4f} +/- {1:.4f} s '.format(
        times.mean(), times.std()))
    acs.show(filename='figure/ACO_10_cities.png')
    acs.plot(filename='figure/ACO_10_cities_curve.png')


# %%
