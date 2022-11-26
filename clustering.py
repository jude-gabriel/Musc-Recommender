from sklearn.cluster import KMeans
import numpy as np

def kmeans(train, test):
    # Run kmeans on the training set
    # Todo: how many centroids????
    kmeans = KMeans(n_clusters=2, random_state=0).fit((train[:, 0:len(train[0]) - 2]).astype(float))

    # Get the list of centroids
    centroids = kmeans.cluster_centers_
    print(kmeans.labels_)

    # Measure each clusters density
    # Predict each point in training set
    count = np.zeros(centroids[:, 0].size)
    for t in train:
        center = kmeans.predict(np.array([(t[0:len(t)-2]).astype(float)]))
        print(center.shape)
        print(center)
        idx = find_index(center, centroids)
        count[idx] = count[idx] + 1

    # Count how many points per each centroid. Most dense will have most points
    most_dense = centroids[np.argmax(count)]
    print(most_dense.shape)
    print(most_dense)


    # Predict each item in test set
    recommend = np.array([])
    for t in test:
        # If it's centroid is the centroid of the most dense cluster, add it to songs to recommend
        center = kmeans.predict(np.array([(t[0:len(t) - 2]).astype(float)]))
        print(center.shape)
        print(center)
        if np.array_equal(center, most_dense):
            print(True)
            recommend = np.append(recommend, t[len(t)-2])

    return recommend


def find_index(point, centroids):
    for i in range(centroids[:, 0].size):
        if np.array_equal(point, centroids[i]):
            print(i)
            return i