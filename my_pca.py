import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition


def main(dataset):
    num = dataset.shape[0]

    dataset_centered = dataset - dataset.mean(axis=0)

    plt.figure()
    plt.scatter(dataset_centered[:, 0], dataset_centered[:, 1], c='blue')
    plt.show()

    cov = np.cov(dataset_centered, rowvar=False)

    vals, vecs = np.linalg.eig(cov)
    vect_max = vecs[np.argmax(vals)].reshape(2, -1)

    print(vect_max[0]**2+vect_max[1]**2)

    dataset_new = np.dot(dataset_centered, vect_max)

    plt.figure()
    plt.scatter(dataset_new[:, 0], [0 for i in range(num)], c='red')
    plt.show()

def main_2():
    num = 15
    dataset = np.array([[i, 2 * i + np.random.uniform(-3, 3), np.random.uniform(0, 20)] for i in range(num)])

    fig = plt.figure()
    ax = Axes3D(fig, elev=48, azim=134)
    plt.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c='black')
    plt.show()

    dataset_centered = dataset - dataset.mean(axis=0)

    cov = np.cov(dataset_centered, rowvar=False)

    vals, vecs = np.linalg.eig(cov)
    eig_pairs = [(vals[i], vecs[i]) for i in range(len(vals))]
    eig_pairs.sort()
    eig_pairs.reverse()

    dataset_new = np.hstack((eig_pairs[0][1].reshape(3, 1),
                          eig_pairs[1][1].reshape(3, 1)))

    Y = dataset_centered.dot(dataset_new)

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c='red')
    plt.show()


if __name__ == '__main__':
    main_2()
