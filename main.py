import feature_loader as fl
import clustering as cl
import genre_analysis as ga

# Get the train and test data
train, test = fl.getFeatures()

# Run k-means to get list of songs to recommend
initial_clusters, initial_labels, final_clusters, final_labels = cl.kmeans(train, test)

# Do genre analysis to determine before and after sub-genres
ga.analysis(initial_clusters, initial_labels, final_clusters, final_labels)
