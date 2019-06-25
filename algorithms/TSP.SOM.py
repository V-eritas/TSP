# %%
import time
import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib as mpl
# %%


class SOM:
    '''Self-Organising Map for TSP'''

    def __init__(self, cities, iterations=10000, learning_rate=0.8):
        # cities:pd.DataFrame 城市列表,每个城市一行
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.cities = cities
        # 设置初始邻域范围
        self.radius = self.cities.shape[0] * 8
        # 随机生成网络
        self.network = self.__generate_network()
        self.distance_curve = []

    def __generate_network(self):
        '''生成指定数量的2维向量，每个维度的值在[0,1],作为网络初始值'''
        return np.random.rand(self.radius, 2)

    def __get_neighborhood(self, center, radix, domain):
        """生成指定半径,中心的高斯分布"""
        # 避免方差过小，出现NaN
        if radix < 1:
            radix = 1
        deltas = np.absolute(center - np.arange(domain))
        distances = np.minimum(deltas, domain - deltas)
        # 计算指定的高斯分布
        return np.exp(-(distances*distances) / (2*(radix*radix)))

    def __select_closest(self, city):
        '''选择离city最近的神经元'''
        return (np.linalg.norm(self.network - city, axis=1)).argmin()

    def __get_path(self):
        """返回网络产生的路径"""
        self.cities['winner'] = self.cities[['x', 'y']].apply(
            lambda c: self.__select_closest(c), axis=1, raw=True)
        return (self.cities.sort_values('winner').index - 1).values.tolist()

    def __path_distance(self, path):
        points = self.cities[['x', 'y']].iloc[path,:]
        distances = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
        return np.sum(distances)

    def run(self):
        for i in range(self.iterations):
            if not i % 100:
                print('\t> Iteration {0}/{1}'.format(i, self.iterations), end="\r")
            # 随机选择一个城市
            city = self.cities.sample(1)[['x', 'y']].values
            # 选择网络中距离最近的神经元
            winner_idx = self.__select_closest(city)
            # 生成作用于优胜邻域的函数
            gaussian = self.__get_neighborhood(
                winner_idx, self.radius//10, self.network.shape[0])
            # 更新网络权重，使神经元更靠近城市位置
            self.network += gaussian[:, np.newaxis] * \
                self.learning_rate * (city - self.network)
            # 学习率，半径衰减
            self.learning_rate = self.learning_rate * 0.99997
            self.radius = self.radius * 0.9997

            # for plotting dsitance curve
            self.path = self.__get_path()
            self.distance = self.__path_distance(self.path)
            self.distance_curve.append(self.distance)

            if not i % 1000:
                self._plot_network(name=r'figure/SOM{:05d}.png'.format(i))

            if self.radius < 1:
                print('Radius has completely decayed, finishing execution at {0} iterations'.format(i))
                break
            if self.learning_rate < 0.001:
                print('Learning rate has completely decayed, finishing execution at {0} iterations'.format(i))
                break
        else:
            print('{} iterations completed'.format(self.iterations))

        self._plot_network(name=r'figure/final.png')
        self.path = self.__get_path()
        self.distance = self.__path_distance(self.path)
        self.path.append(self.path[0])
        self._plot_path(r'figure/path.png')
        return self.path, self.distance

    def _plot_network(self, name='diagram.png'):
        fig = plt.figure(figsize=(5, 5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])
        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')
        axis.scatter(self.cities['x'], self.cities['y'], color='red', s=4)
        axis.plot(self.network[:, 0], self.network[:, 1],
                    'r.', ls='-', color='#66aacc', markersize=2)
        plt.savefig(name)
        plt.close()

    def _plot_path(self, name='diagram.png'):
        fig = plt.figure(figsize=(5, 5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])
        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')
        axis.scatter(self.cities['x'], self.cities['y'], color='red', s=4)
        best_path = self.cities.iloc[self.path,:]
        # self.path.loc[self.path.shape[0]] = self.path.iloc[0]
        axis.plot(best_path['x'], best_path['y'], color='b', linewidth=1)
        plt.savefig(name)
        plt.close()

    def plot(self, name='curve.png'):
        plt.figure(figsize=(5,5))
        plt.plot(range(len(self.distance_curve)), self.distance_curve)
        plt.show()
        plt.savefig(r'figure/'+name)
# %%

if __name__ == '__main__':
    nodes = pd.read_csv(r'data\TSP100cities.tsp', sep=' ', names=[
                        'ID', 'x', 'y'], header=None, index_col=0)
    times = np.ones(1)
    i = 0
    while i < len(times):
        s = SOM(nodes, 16000, learning_rate=0.8)
        start = time.clock()
        path, mincost = s.run()
        finish = time.clock()
        times[i] = (finish - start)
        i = i + 1
    print('best path is: {0}, with total_cost = {1:6f}'.format(
        ' -> '.join(str(i) for i in path), mincost))
    print('The average running time is {0:.4f} +/- {1:.4f} s '.format(
        times.mean(), times.std()))
    s.plot(r'figure/GA_10_citeis_curve.png')
# %%
