import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage



def analysis(initial_clusters, initial_labels, final_clusters, final_labels):
    for i in range(len(initial_clusters)):
        # Initial Clusters: Plot the dendrogram to observe the sub clusters
        matrix = linkage(initial_clusters[i], method='ward')
        label = [x.rsplit('/', 1)[-1] for x in initial_labels[i]]
        dendro = dendrogram(matrix, labels=label)
        plt.title("Initial Dendrogram: Cluster " + str(i))
        plt.ylabel("Euclidean Distance")
        plt.show()

        # Final Clusters: Plot the dendrogram to observe the sub clusters
        matrix = linkage(final_clusters[i], method='ward')
        label = [x.rsplit('/', 1)[-1] for x in final_labels[i]]
        dendro = dendrogram(matrix, labels=label)
        plt.title("Final Dendrogram: Cluster " + str(i))
        plt.ylabel("Euclidean Distance")
        plt.show()

