import numpy as np
from sklearn.cluster import KMeans

# X = np.array([[1, 2], [1, 4], [1, 0],
#               [4, 2], [4, 4], [4, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# print(kmeans.labels_)

# print(kmeans.predict([[0, 0], [4, 4]]))

# print(kmeans.cluster_centers_)

# print(kmeans.cluster_centers_[kmeans.predict(X)])

# -----------------------------------------------------

from sklearn.datasets import make_blobs, make_circles, load_iris
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#ab45cd', '#abcdef'])
# X_1, _ = make_circles(300, factor=0.5, noise=0.08)
# X_1, _ = make_blobs(400, centers=5)


iris = load_iris()
X = iris.data
y = iris.target


# kmeans = KMeans(n_clusters=5).fit(X_1)
kmeans = KMeans(n_clusters=3).fit(X)
# def_predict = kmeans.fit_predict(X_1)
def_predict = kmeans.fit_predict(X)
# centroid = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)
# plt.scatter(X_1[:, 0], X_1[:, 1], c=colors[def_predict])
plt.scatter(X[:, 0], X[:, 1], X[:, 3], c=colors[def_predict])
# plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=100, c='red')
plt.show()

# inertia = []
# for k in range(1, 8):
#     kmeans = KMeans(n_clusters=k, random_state=1).fit(X_1)
#     inertia.append(np.sqrt(kmeans.inertia_))
# plt.plot(range(1, 8), inertia, marker='s')
# plt.show()
# print(plt.xlabel('$k$'))
# print(plt.ylabel('$J(C_k)$'))

