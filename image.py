from matplotlib.image import imread, imsave
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def get_img(path, show=True):
    original = imread(path)
    if show:
        plt.imshow(original)
        plt.show()
        print('shape: ', original.shape)
    return original


def get_kmeans(original, n_colors=8):
    X = original.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors).fit(X)
    pred = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    return centroids[pred].reshape(original.shape)

if __name__ == '__main__':
    n_colors = 8
    images = []
    original = get_img('spiderman.png', False)
    new = get_kmeans(original, n_colors)
    images.append(original)
    images.append(new)

    fig, axarr = plt.subplots(1, 2, sharex=True, figsize=(20,8))
    axarr[0].imshow(images[0])
    axarr[1].imshow(images[1])
    fig.tight_layout()
    plt.show()