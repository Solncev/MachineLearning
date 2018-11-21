from math import sqrt

import numpy as np
from matplotlib import pyplot as plt


class K_Means:
    def __init__(self, data, n_clusters=3, method=None):
        self.data = data
        self.n_clusters = n_clusters
        self.max_iter = 10
        self.tolerance = .01
        self.fitted = False
        self.labels = np.array([])
        # self.centroids = np.array(self.data[:n_clusters], dtype='f')
        x = np.copy(self.data)
        np.random.shuffle(x)
        self.centroids = np.array(x[:n_clusters], dtype='f')
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
        self.labels = np.array([])
        for elem in self.data:
            dist2 = [self.method(elem, center) for center in self.centroids]
            self.labels = np.append(self.labels.tolist(), dist2.index(min(dist2))).astype(int)

    def recalculate_centroids(self):
        for i in range(self.n_clusters):
            num = 0
            temp = np.zeros(self.data[0].shape)
            for k, label in enumerate(self.labels):
                if label == i:
                    temp += self.data[k]
                    num += 1
            self.centroids[i] = temp / num

    def fit(self):
        iter = 1
        while iter < self.max_iter:
            prev_centroids = np.copy(self.centroids)
            self.distribute_data()
            self.recalculate_centroids()
            if max([self.method(i, k) for i, k in zip(self.centroids, prev_centroids)]) < self.tolerance:
                break
            iter += 1
        self.fitted = True

    def predict(self, data):
        if self.fitted:
            dist2 = [self.method(data, center) for center in self.centroids]
            return dist2.index(min(dist2))


def kinda_main(l1, k=3, method=''):
    test = K_Means(l1, k, method)
    test.fit()

    if (k < 10) & (test.data.shape[1] == 2):
        colors = np.array(
            ['#377eb8', '#ff7f00', '#4daf4a', '#ab45cd', '#abcdef', '#789654', '#abcdef', '#123654', '#deca57',
             '#f5e16a'])
        plt.figure()
        plt.scatter(test.data[:, 0], test.data[:, 1], c=colors[test.labels])
        plt.scatter(test.centroids[:, 0], test.centroids[:, 1], marker='x', s=100)
        plt.show()


if __name__ == '__main__':
    clusters = 3
    method = 'first'
    game_flag = False
    l1 = np.array([[1, 2, 3], [3, 4, 6], [5, 6, 1], [4, 3, 2], [4, 6, 3], [3, 2, 0]])
    kinda_main(l1, clusters, method)
