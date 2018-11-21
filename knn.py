import numpy as np


class knn:
    def __init__(self, dataset, k_num = 3):
        self.dataset = dataset
        self.k_num = k_num

    def get_dist(self, list1, list2):
        return np.sqrt(sum((i-j)**2 for i, j in zip(list1, list2)))

    def predict(self, data):
        dist = np.array([[self.get_dist(data, i[0]), i[1][0]] for i in self.dataset])
        sort = dist[dist[:, 0].argsort()][:self.k_num]
        counts = {}
        for s in sort:
            if int(s[1]) not in counts:
                counts[int(s[1])] = 0
            counts[int(s[1])] += 1
        return max(counts.keys(), key = lambda cl: counts[cl])

def main(dataset):
    test = knn(dataset, 5)
    test.predict([120, 120])

if __name__ == '__main__':
    pass