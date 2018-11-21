from math import sqrt

import numpy as np
from matplotlib import pyplot as plt


class C_Means:
    def __init__(self, data, n_clusters=3, method=None, m=2, cut=.5):
        self.data = data
        self.n_clusters = n_clusters
        self.max_iter = 100
        self.tolerance = .01
        self.labels = np.array([])
        self.m = m
        self.cut = cut
        self.dist = np.zeros((self.data.shape[0], self.n_clusters))
        self.centroids = np.zeros((self.n_clusters, self.data.shape[1]))    
        self.u = np.array([[np.random.uniform(0, 1) for i in range(self.n_clusters)] for j in range(self.data.shape[0])])
        if method == 'first':
            self.method = self.sqrt_dist
        elif method == 'third':
            self.method = self.abs_dist
        elif method == 'fourth':
            self.method = self.max_abs_dist
        else:
            self.method = self.basic_dist

    def basic_dist(self, list1, list2):
        return sum((i - j) ** 2 for i, j in zip(list1, list2))

    def sqrt_dist(self, list1, list2):
        return sqrt(self.basic_dist(list1, list2))

    def abs_dist(self, list1, list2):
        return sum(abs(i - j) for i, j in zip(list1, list2))

    def max_abs_dist(self, list1, list2):
        return max(abs(i - j) for i, j in zip(list1, list2))

    def distribute_data(self):
        self.dist = np.array(
            [[self.method(i, j) for i in self.centroids] for j in self.data])  # Расстояния до центроидов
        self.u = (1 / self.dist) ** (1 / (self.m - 1))  # Принадлежности кластерам
        self.u = (self.u / self.u.sum(axis=1)[:, None])

    def recalculate_centroids(self):
        self.centroids = (self.u.T).dot(self.data) / self.u.sum(axis=0)[:, None]

    def fit(self):
        iter = 1
        while iter < self.max_iter:
            prev_centroids = np.copy(self.centroids)
            self.recalculate_centroids()
            self.distribute_data()
            if max([self.method(i, k) for i, k in zip(self.centroids, prev_centroids)]) < self.tolerance:
                break
            iter += 1
        self.fitted = True

    def predict(self, data):
        if self.fitted:
            dist2 = [self.method(data, center) for center in self.centroids]
            return dist2.index(min(dist2))

    def get_labels(self):
        u_max = self.u.max(axis=1)
        labels = np.array([np.where(self.u[k, :] == u_max[k]) for k in range(self.u.shape[0])]).reshape(1, -1)
        for i, elem in enumerate(u_max):
            if elem < self.cut:
                labels[0, i] = self.n_clusters + 1
        return labels[0, :]


def kinda_main(l1, k=3, method='', cut=0.9):
    test = C_Means(l1, k, method, cut=cut)
    test.fit()
    print(test.u)

    if (k < 10) & (test.data.shape[1] == 2):
        colors = np.array(
            ['#377eb8', '#ff7f00', '#4daf4a', '#ab45cd', '#abcdef', '#789654', '#abfeef', '#123654', '#deca57',
             '#f5e16a'])
        plt.figure()
        plt.scatter(test.data[:, 0], test.data[:, 1], c=colors[test.get_labels()])
        plt.scatter(test.centroids[:, 0], test.centroids[:, 1], marker='x', s=100)
        plt.show()


def basic_dist(list1, list2):
    return sum((i - j) ** 2 for i, j in zip(list1, list2))


if __name__ == '__main__':
    clusters = 3
    method = 'first'
    game_flag = False
    l1 = np.array([[1, 2, 3], [3, 4, 6], [5, 6, 1], [4, 3, 2], [4, 6, 3], [3, 2, 0]])
    # l1 = np.array([[2, 3], [4, 6], [6, 1], [4, 2], [4, 5], [3, 0]])
    kinda_main(l1, clusters, method)

    # a1 = np.array([[np.random.uniform(0, 20), np.random.uniform(0, 20)] for k in range(1, 8)])
    # print(a1)
    # num = 3
    # a2 = a1[np.random.choice(a1.shape[0], size=num, replace=False)]
    # a2 = a2 + 0.0000001
    # print(a2)
    # a3 = np.array([[basic_dist(i, j) for i in a2] for j in a1])
    # print(a3)
    #
    # m = 1.1
    # np.seterr(divide='ignore')
    # u = (1 / a3) ** (1 / (m - 1))
    # print(u)
    # um = (u / u.sum(axis=1)[:, None])
    # print(um)
    #
    # x = (um.T).dot(a1) / um.sum(axis=0)[:, None]
    # print(x)
