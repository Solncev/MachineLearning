import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import decomposition
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def get_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


def main():
    X, y = get_data()

    fig = plt.figure(1, figsize=(6, 5))
    ax = Axes3D(fig, elev=48, azim=134)

    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Verginica', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name)
    y_clr = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_clr, cmap=plt.cm.nipy_spectral)

    plt.show()


def main_2():
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42)
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba((X_test))

    print('Accuracy: {:.5f}'.format(accuracy_score(y_test,
                                                   preds.argmax(axis=1))))

    pca = decomposition.PCA(n_components=2)
    X_centered = X - X.mean(axis=0)
    pca.fit(X_centered)
    X_pca = pca.transform(X_centered)

    plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
    plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
    plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
    plt.legend(loc=0)
    plt.show()


def main_3():
    X, y = get_data()

    pca = decomposition.PCA(n_components=2)
    X_centered = X - X.mean(axis=0)
    pca.fit(X_centered)
    X_pca = pca.transform(X_centered)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=.3,
                                                        random_state=42)
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba((X_test))

    print('Accuracy: {:.5f}'.format(accuracy_score(y_test,
                                                   preds.argmax(axis=1))))

    plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
    plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
    plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    # main()
    main_2()
    main_3()
