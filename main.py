import feature_loader as fl
import clustering as cl
import music_loader as ml


# Get the train and test data
#Checks if feature text file exists,
#if not creates file from wav

if(ml.is_loaded() == False):
    ml.load_wav()
   
train, test = fl.getFeatures()


# Run k-means
songs_to_recommend = cl.kmeans(train, test)
print(songs_to_recommend)

