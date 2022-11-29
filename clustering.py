from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

def kmeans(train, test):
    # Get the features
    X = train[:, 0:len(train[0]) - 1].astype(float)
    X_labels = train[:, len(train[0])-1]
    Y = test[:, 0:len(train[0]) - 1].astype(float)
    Y_labels = test[:, len(test[0]) - 1]

    print(X_labels)
    print(Y_labels)

    # Run kmeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    labels = kmeans.fit_predict(X)
    
    plotClusters(X,labels)
    
    # Get the list of centroids
    centroids = kmeans.cluster_centers_
    print(kmeans.labels_)

    # Measure each clusters density
    # Predict each point in training set
    count = np.zeros(centroids[:, 0].size)
    for x in X:
        center = kmeans.predict(np.array([x]))
        print(center)
        count[center] = count[center] + 1

    # Count how many points per each centroid. Most dense will have most points
    most_dense = np.argmax(count)
    print(most_dense)

    # Predict each item in test set
    recommend = np.array([])

    # Predict each song in the test split. If it is in the most dense cluster, add to songs to recommend
    for i in range(len(Y_labels)):
        center = kmeans.predict(np.array([Y[i]]))
        print(center)
        if center == most_dense:
            recommend = np.append(recommend, Y_labels[i])


    return recommend


def plotClusters(X, labels):
    plt.scatter(X[:,0],X[:,1])

    #Getting unique labels
    u_labels = np.unique(labels)
 
#plotting the results:
    for i in u_labels:
        plt.scatter(X[labels == i , 0] , X[labels == i , 1] , label = i)
    plt.legend()
    plt.show()
    