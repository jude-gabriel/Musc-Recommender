import feature_loader as fl
import clustering as cl


# Get the train and test data
train, test = fl.getFeatures()

# Run k-means
songs_to_recommend = cl.kmeans(train, test)
print(songs_to_recommend)

#ok
