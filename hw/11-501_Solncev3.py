from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def main():
    num = 15
    dataset = np.array([[i, 2 * i + np.random.uniform(-3, 3)] for i in range(num)])

    plt.figure()
    plt.scatter(dataset[:, 0], dataset[:, 1], c='black')
    plt.show()

    print(dataset.mean(axis=0))

    dataset_centered = dataset - dataset.mean(axis=0)

    plt.figure()
    plt.scatter(dataset_centered[:, 0], dataset_centered[:, 1], c='blue')
    plt.show()

    covmat = np.cov(dataset_centered, rowvar=False)

    vals, vects = np.linalg.eig(covmat)
    print(np.argmax(vals))
    vect_max = vects[np.argmax(vals)]
    print(vect_max.shape)  # размерность вектора
    print(dataset.shape)  # размерность dataset

    vect_max = vect_max.reshape(2, -1)  # поменяли размерность
    dataset_new = np.dot(dataset_centered, vect_max)

    print(vect_max[0] ** 2)

    plt.figure()
    plt.scatter(dataset_new[:, 0], [0 for i in range(num)], c='red')
    plt.show()

    # HOMEWORK
    num = 15
    dataset3 = np.array([[i, 2 * i + np.random.uniform(-3, 3), np.random.uniform(0, 50)] for i in range(num)])

    fig = plt.figure()
    ax = Axes3D(fig, elev=48, azim=134)
    plt.scatter(dataset3[:, 0], dataset3[:, 1], dataset3[:, 2], c='green')
    plt.show()



#HOMEWORK
def for_two_features():
    num = 15
    size=20

    dataset = np.array([[i, 2 * i + np.random.uniform(-3, 3), np.random.uniform(0, size)] for i in range(num)])

    fig = plt.figure()
    axes = Axes3D(fig, elev=48, azim=134)
    plt.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c='black')
    plt.show()

    dataset_centered = dataset - dataset.mean(axis=0)
    covariance = np.cov(dataset_centered, rowvar=False)

    values, vectors = np.linalg.eig(covariance)
    eigenvalues_pairs = [(values[i], vectors[i]) for i in range(len(values))]
    eigenvalues_pairs.sort()
    eigenvalues_pairs.reverse()

    new_dataset = np.hstack((eigenvalues_pairs[0][1].reshape(3, 1), eigenvalues_pairs[1][1].reshape(3, 1)))
    Y = dataset_centered.dot(new_dataset)

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c='yellow')
    plt.show()


if __name__ == '__main__':
    for_two_features()
