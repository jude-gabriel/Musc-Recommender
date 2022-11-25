import feature_loader as fl
import clustering as cl

# Get the train and test data
train, test = fl.getFeatures()


# Run k-means
cl.kmeans(train, test)
