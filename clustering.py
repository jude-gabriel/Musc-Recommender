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
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, init='k-means++').fit(X)

    # Get the labels and plot the initial clusters
    labels = kmeans.fit_predict(X)
    inital_data, initial_labels = plotClusters(X, X_labels, labels, num_clusters, get_labels("INITIAL_CLUSTERS", labels, 0))

    # Measure each clusters density
    # Predict each point in training set
    count = np.zeros(kmeans.cluster_centers_[:, 0].size)
    for x in X:
        center = kmeans.predict(np.array([x]))
        count[center] = count[center] + 1

    # Count how many points per each centroid. Most dense will have most points
    most_dense = np.argmax(count)
    print("Most Dense Cluster: Cluster " + str(most_dense))

    # Get the points in the most dense cluster
    most_dense_points = get_most_most_dense_points(most_dense, labels, X)

    # Predict each song in the test split. If it is in the most dense cluster, add to songs to recommend
    # Get labels for graphing final clusters and songs to recommend
    recommend = np.array([])
    test_labels = np.array([])
    recommend_labels = np.array([])
    for i in range(len(Y_labels)):
        center = kmeans.predict(np.array([Y[i]]))
        songs_centroid = kmeans.cluster_centers_[center]

        # Get the centroid for the centers label
        # Get the most dense cluster
        # Get the points in the most dense cluster
        if nearest_neighbors(Y[i], center, songs_centroid, most_dense, most_dense_points):
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
    print("Songs to Recommend:", recommend)

    # Get return initial and final cluster for sub-genre analysis
    final_data, final_labels = get_final_clusters(X, Y, X_labels, Y_labels, test_labels, num_clusters)
    return inital_data, initial_labels, final_data, final_labels


def get_final_clusters(X, Y, X_labels, Y_labels, which_cluster, num_clusters):
    data = np.append(X, Y, axis=0)
    labels = np.append(X_labels, Y_labels)
    cluster_data = []
    cluster_labels = []
    for i in range(num_clusters):
        clust_data = []
        clust_labels = []
        for j in range(which_cluster.size):
            if int(which_cluster[j]) == i:
                clust_data.append(data[j])
                clust_labels.append(labels[j])
        cluster_data.append(clust_data)
        cluster_labels.append(clust_labels)
    return cluster_data, cluster_labels


def get_most_most_dense_points(most_dense, labels, points):
    most_dense_points = np.array([])
    count = 0
    for i in range(labels.size):
        if labels[i] == most_dense:
            count = count + 1
            most_dense_points = np.append(most_dense_points, points[i])
    most_dense_points = most_dense_points.reshape(count, int(most_dense_points.size / count))
    return most_dense_points


def nearest_neighbors(song, songs_label, songs_centroid, most_dense_label, most_dense_points):
    # If it is classified in the most dense centroid then skip. Already recommended
    if songs_label == most_dense_label:
        print("Added by label")
        return True
    # Then measure the distance from the point to its centroid
    dist_to_centroid = np.linalg.norm(song - songs_centroid)
    # Then measure the distance to points in most dense cluster
    dist_to_dense_points = np.array([])
    for i in range(most_dense_points[:, 0].size):
        dist_to_dense_points = np.append(dist_to_dense_points, np.linalg.norm(song - most_dense_points[i]))

    # If any distance to most dense cluster is smaller than its own centroid return true
    if dist_to_centroid > np.min(dist_to_dense_points):
        print("Added by distance factor")
        return True
    print("Should not be added")
    return False


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
    clusters = []
    clust_labels = []
    for i in range(num_clusters):
        songs = np.array([])
        clust = []
        l = []
        for j in range(X[:, 0].size):
            if labels[j] == i:
                clust.append(X[j])
                l.append(X_labels[j])
                songs = np.append(songs, X_labels[j])
        print("Cluster " + str(i) + ": ", songs)
        clusters.append(clust)
        clust_labels.append(l)

    # Plot the clusters with training data
    plotResults(X, plot_labels, "Initial Clusters")
    return clusters, clust_labels


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

