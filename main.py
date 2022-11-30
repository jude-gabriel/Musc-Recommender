import feature_loader as fl
import clustering as cl
import music_loader as ml
import matplotlib.pyplot as plt
# Get the train and test data
#Checks if feature text file exists,
#if not creates file from wav

   
train, test = fl.getFeatures()

# Run k-means to get list of songs to recommend
songs_to_recommend = cl.kmeans(train, test)
print("Songs to Recommend:", songs_to_recommend)

plt.show()

