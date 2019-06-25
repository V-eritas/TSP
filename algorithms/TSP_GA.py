import pandas as pd
import numpy as np
import time
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

SCORE_NONE = -1

class Life(object):
    def __init__(self, Gene=None):
        self.gene = Gene
        self.score = SCORE_NONE
    def show(self):
        print(' -> '.join(str(i) for i in self.gene))

class Ga(object):
    def __init__(self, CrossRate, MutationRage, LifeCount, GeneLenght, MatchFun=lambda life: 1):
        self.croessRate = CrossRate
        self.mutationRate = MutationRage
        self.lifeCount = LifeCount
        self.geneLenght = GeneLenght
        self.matchFun = MatchFun
        self.lives = []
        self.best = None                          # 保存一代中最优个体
        self.generation = 0
        self.crossCount = 0
        self.mutationCount = 0
        self.bounds = 0.0                         # 用于计算选择概率
        self.initPopulation()

    def initPopulation(self):
        self.lives = []
        for i in range(self.lifeCount):
            gene = [x for x in range(self.geneLenght)]
            random.shuffle(gene)
            life = Life(gene)
            self.lives.append(life)

    def judge(self):
        self.bounds = 0.0
        self.best = self.lives[0]
        for life in self.lives:
            life.score = self.matchFun(life)
            self.bounds += life.score
            if self.best.score < life.score:
                self.best = life

    def cross(self, parent1, parent2):
        index1 = random.randint(0, self.geneLenght - 1)
        index2 = random.randint(index1, self.geneLenght - 1)
        tempGene = parent2.gene[index1:index2]
        newGene = []
        p1len = 0
        for g in parent1.gene:
            if p1len == index1:
                newGene.extend(tempGene)
                p1len += 1
            if g not in tempGene:
                newGene.append(g)
                p1len += 1
        self.crossCount += 1
        return newGene

    def mutation(self, gene):
        index1 = random.randint(0, self.geneLenght - 1)
        index2 = random.randint(0, self.geneLenght - 1)

        newGene = gene[:]
        newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
        self.mutationCount += 1
        return newGene

    def getOne(self):
        r = random.uniform(0, self.bounds)
        for life in self.lives:
            r -= life.score
            if r <= 0:
                return life
        raise Exception("选择错误", self.bounds)

    def newChild(self):
        parent1 = self.getOne()
        rate = random.random()

        # 交叉
        if rate < self.croessRate:
            parent2 = self.getOne()
            gene = self.cross(parent1, parent2)
        else:
            gene = parent1.gene

        # 突变
        rate = random.random()
        if rate < self.mutationRate:
            gene = self.mutation(gene)

        return Life(gene)

    def next(self):
        self.judge()
        newLives = []
        newLives.append(self.best)      # 把最好的个体加入下一代
        while len(newLives) < self.lifeCount:
            newLives.append(self.newChild())
        self.lives = newLives
        self.generation += 1

class TSP_GA():
    def __init__(self, file, iterations):
        self.iterations = iterations
        self.cities = pd.read_csv(file, sep=' ', header=None, index_col=0)
        self.dist = cdist(self.cities, self.cities, 'Euclidean')
        self.lifeCount = self.cities.shape[0]
        self.ga = Ga(CrossRate=0.7,
                     MutationRage=0.02,
                     LifeCount=self.lifeCount,
                     GeneLenght=len(self.cities),
                     MatchFun=self.matchFun())
        self.distance_curve = np.zeros(iterations)

    def distance(self, order):
        distance = 0.0
        for i in range(-1, len(self.cities) - 1):
            index1, index2 = order[i], order[i + 1]
            distance += self.dist[index1][index2]
        return distance

    def matchFun(self):
        return lambda life: 1.0 / self.distance(life.gene)

    def solve(self):
        iter =0
        while iter <self.iterations:
            self.ga.next()
            distance = self.distance(self.ga.best.gene)
            self.distance_curve[iter] = distance
            # print("{0:3d}:  {1:f}".format(self.ga.generation, distance))
            if  iter%10==0:
                # self.show()
                pass
            iter += 1
            # self.plot() 
        self.path = self.ga.best.gene
        distance = self.distance(self.ga.best.gene)
        return self.path, distance

    def curve(self):
        plt.figure(figsize=(5,5))
        plt.plot(range(self.iterations), self.distance_curve, c='r')
        plt.show()
        plt.gcf().clear()

    def show(self):
        life = self.ga.best
        life.show()

    def plot(self,filename='GA_100_cities.png', annotation_size=8, dpi=200, save=True):
        plt.figure(figsize=(5,5))
        x = self.cities.iloc[self.ga.best.gene,0]
        y = self.cities.iloc[self.ga.best.gene,1]
        plt.plot(x, y, c='r',label="path")
        plt.scatter(x, y, s=4)
        plt.title('Genetic Algorithm for TSP')
        for i in self.ga.best.gene:
            plt.annotate(i, xy = self.cities.iloc[i,:], xytext=self.cities.iloc[i,:]+(4,4) , size=annotation_size)
        if save:
            plt.savefig(r'figure/'+filename, dpi=dpi)
        plt.show()
        plt.gcf().clear()

def main():
    times = np.ones(10)
    i = 0
    while i < len(times):
        tsp = TSP_GA(r'data\TSP100cities.tsp', 4000)
        start = time.clock()
        path, mincost = tsp.solve()
        finish = time.clock()
        times[i] = (finish - start)
        i = i + 1
    path.append(path[0])
    print('best path is: {0}, with total_cost = {1:6f}'.format(
            ' -> '.join(str(i) for i in path), mincost))
    print('The average running time is {0:.4f} +/- {1:.4f} s '.format(
        times.mean(), times.std()))
    tsp.plot()
    tsp.curve()

if __name__ == '__main__':
    main()
