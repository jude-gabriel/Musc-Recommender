import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min


def getMinDistances(train, test, most_dense):
    # Get the features
    X = train[:, 0:len(train[0]) - 1].astype(float)
    X_labels = train[:, len(train[0]) - 1]
    Y = test[:, 0:len(train[0]) - 1].astype(float)
    Y_labels = test[:, len(test[0]) - 1]

    # bring down to 2d
    X = TSNE(n_components=2, perplexity=3, init='random', learning_rate='auto').fit_transform(X)
    Y = TSNE(n_components=2, perplexity=3, init='random', learning_rate='auto').fit_transform(Y)

    # Run kmeans
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)

    # get give closest to cluster centers for each cluster, we only care about most_dense
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, Y)
    outputFiveIndex = np.zeros(5)
    outputFiveIndex[0] = closest[most_dense]
    oldY = Y
    Y = np.delete(Y, int(outputFiveIndex[0]), axis=0)

    # then produce 5 closest to that center x,y coordinate from Y data
    for i in range(len(outputFiveIndex)):
        if i != 0:
            index = int(outputFiveIndex[i-1])
            closest, _ = pairwise_distances_argmin_min(np.array([oldY[index]]), Y)
            Y = np.delete(Y, int(closest[most_dense]), axis=0)
            outputFiveIndex[i] = closest[most_dense]

    outputLabels = np.empty(len(outputFiveIndex), dtype = "S512")
    for i in range(len(outputFiveIndex)):
        outputLabels[i] = Y_labels[int(outputFiveIndex[i])]

    # print recommended songs
    print("Songs to recommond based off min-distance: ", outputLabels)
    plotEverything(X, oldY, outputFiveIndex, kmeans.cluster_centers_, most_dense)

# plot inital points, cluster that was picked, and ending choices
def plotEverything(X, Y, outputFiveIndex, centers, most_dense):

    plt.title("Initial Testing Data and Centroid")
    plt.scatter(X[:, 0], X[:, 1], color='blue')
    plt.scatter(centers[most_dense][0], centers[most_dense][1], color='orange')
    plt.show()

    plt.title("Testing Data and Centroid")
    plt.scatter(Y[:, 0], Y[:, 1], color='blue')
    plt.scatter(centers[most_dense][0], centers[most_dense][1], color='orange')
    plt.show()

    plt.title("Testing Data, Centroid, Songs Chosen")
    plt.scatter(Y[:, 0], Y[:, 1], color='blue')
    plt.scatter(centers[most_dense][0], centers[most_dense][1], color='orange')
    for i in range(len(outputFiveIndex)):
        plt.scatter(Y[int(outputFiveIndex[i])][0], Y[int(outputFiveIndex[i])][1], color='red')
    plt.show()