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
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)

    # Get the labels and plot the initial clusters
    labels = kmeans.fit_predict(X)
    plotClusters(X, X_labels, labels, num_clusters, get_labels("INITIAL_CLUSTERS", labels, 0))

    # Measure each clusters density
    # Predict each point in training set
    count = np.zeros(kmeans.cluster_centers_[:, 0].size)
    for x in X:
        center = kmeans.predict(np.array([x]))
        count[center] = count[center] + 1

    # Count how many points per each centroid. Most dense will have most points
    most_dense = np.argmax(count)
    print("Most Dense Cluster: Cluster " + str(most_dense))

    recommend = np.array([])

    # Predict each song in the test split. If it is in the most dense cluster, add to songs to recommend
    # Get labels for graphing final clusters and songs to recommend
    recommend = np.array([])
    test_labels = np.array([])
    recommend_labels = np.array([])
    for i in range(len(Y_labels)):
        center = kmeans.predict(np.array([Y[i]]))
        if center == most_dense:
            recommend = np.append(recommend, Y_labels[i])
            recommend_labels = np.append(recommend_labels, -1)
        else:
            recommend_labels = np.append(recommend_labels, center)
        test_labels = np.append(test_labels, center)

    # Plot final clusters and songs to recommend
    test_labels = np.append(labels, test_labels)
    recommend_labels = np.append(labels, recommend_labels)
    data = np.append(X, Y, axis=0)
    plotResults(data, get_labels("TEST_RESULTS", test_labels, most_dense), "Clusters with Test Songs")
    plotResults(data, get_labels("RECOMMEND", recommend_labels, most_dense), "Songs to Recommend")

    # Return list of songs to recommend
    return recommend


def get_labels(plot_type, curr_labels, most_dense):
    labels = np.array([])

    # Get labels for showing different clusters
    if plot_type == "INITIAL_CLUSTERS" or plot_type == "TEST_RESULTS":
        for i in range(curr_labels.size):
            labels = np.append(labels, "Cluster " + str(curr_labels[i]))

    # Get labels for showing which songs to recommend and initial dense cluster
    elif plot_type == "RECOMMEND":
        for i in range(curr_labels.size):
            if curr_labels[i] == most_dense:
                labels = np.append(labels, "Initial Most Dense Cluster")
            elif curr_labels[i] == -1:
                labels = np.append(labels, "Songs to Recommend")
            else:
                labels = np.append(labels, "Other Clusters")
    return labels


def plotClusters(X, X_labels, labels, num_clusters, plot_labels):
    # Get which points belong to each cluster
    for i in range(num_clusters):
        cluster = np.array([])
        for j in range(X[:, 0].size):
            if labels[j] == i:
                cluster = np.append(cluster, X_labels[j])
        print("Cluster " + str(i) + ": ", cluster)

    # Plot the clusters with training data
    plotResults(X, plot_labels, "Initial Clusters")


def plotResults(X, labels, title):
    # Run TSNE for dimensionality reduction
    X_embedded = TSNE(n_components=2, perplexity=3, init='random', learning_rate='auto').fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

    # Getting unique labels
    u_labels = np.unique(labels)

    # plotting the results:
    for i in u_labels:
        plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], label=i)
    plt.legend()
    plt.title(title)
    plt.show()

