from math import sqrt

import numpy as np
from matplotlib import pyplot as plt


# from sklearn.cluster import DBSCAN


class DBSCAN:
    def __init__(self, data, eps=30.0, min_samples=2, method=''):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = 0
        self.clusters = {0: []}
        self.visited = set()
        self.clustered = set()
        self.fitted = False

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

    def region(self, point):
        return [list(q) for q in self.data if self.method(point, q) < self.eps]  # Список списков

    def fit(self):
        for p in self.data:
            if tuple(p) in self.visited:
                continue
            self.visited.add(tuple(p))
            neighbours = self.region(p)
            if len(neighbours) < self.min_samples:
                self.clusters[0].append(list(p))
            else:
                self.n_clusters += 1
                self.expand(p, neighbours)
        self.fitted = True

    def expand(self, point, neighbours):
        if self.n_clusters not in self.clusters:
            self.clusters[self.n_clusters] = []
        self.clusters[self.n_clusters].append(list(point))
        self.clustered.add(tuple(point))
        while neighbours:
            q = neighbours.pop()
            if tuple(q) not in self.visited:
                self.visited.add(tuple(q))
                q_neighbours = self.region(q)
                if len(q_neighbours) > self.min_samples:
                    neighbours.extend(q_neighbours)
                if tuple(q) not in self.clustered:
                    self.clustered.add(tuple(q))
                    self.clusters[self.n_clusters].append(q)
                    if q in self.clusters[0]:
                        self.clusters[0].remove(q)

    def get_labels(self):
        labels = np.array([])
        if not self.fitted:
            self.fit()
        for point in self.data:
            for i in range(self.n_clusters + 1):
                if list(point) in self.clusters[i]:
                    labels = np.append(labels, i).astype(int)
        return labels


def main(data, eps, samp):
    colors = np.array(
        ['#000000', '#ff7f00', '#4daf4a', '#ab45cd', '#abcdef', '#789654', '#abcdef', '#123654', '#deca57',
         '#f5e16a'])

    pred = DBSCAN(data, eps=eps, min_samples=samp, method='first').get_labels()

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=colors[pred])
    plt.show()


if __name__ == '__main__':
    # main()
    a1 = {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
