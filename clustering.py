from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def kmeans(train, test):
    # Get the features
    X = train[:, 0:len(train[0]) - 1].astype(float)
    X_labels = train[:, len(train[0])-1]
    Y = test[:, 0:len(train[0]) - 1].astype(float)
    Y_labels = test[:, len(test[0]) - 1]


    # Run kmeans
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    labels = kmeans.fit_predict(X)
    plotClusters(X, X_labels, labels, num_clusters)

    # Get the list of centroids
    centroids = kmeans.cluster_centers_

    # Measure each clusters density
    # Predict each point in training set
    count = np.zeros(centroids[:, 0].size)
    for x in X:
        center = kmeans.predict(np.array([x]))
        count[center] = count[center] + 1

    # Count how many points per each centroid. Most dense will have most points
    most_dense = np.argmax(count)

    # Predict each item in test set
    recommend = np.array([])

    # Predict each song in the test split. If it is in the most dense cluster, add to songs to recommend
    for i in range(len(Y_labels)):
        center = kmeans.predict(np.array([Y[i]]))
        if center == most_dense:
            recommend = np.append(recommend, Y_labels[i])

    return recommend


def plotClusters(X, X_labels, labels, num_clusters):
    for i in range(num_clusters):
        cluster = np.array([])
        for j in range(X[:, 0].size):
            if labels[j] == i:
                cluster = np.append(cluster, X_labels[j])
        print("Cluster " + str(i) + ": ", cluster)

    X_embedded = TSNE(n_components=2, perplexity=3, init='random', learning_rate='auto').fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

    # Getting unique labels
    u_labels = np.unique(labels)

    # plotting the results:
    for i in u_labels:
        plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], label=i)
    plt.legend()
    plt.show()
