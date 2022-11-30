import feature_loader as fl
import clustering as cl

# Get the train and test data
train, test = fl.getFeatures()

# Run k-means to get list of songs to recommend
songs_to_recommend = cl.kmeans(train, test)
print("Songs to Recommend:", songs_to_recommend)

