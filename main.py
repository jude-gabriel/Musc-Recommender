import feature_loader as fl
import clustering as cl
import music_loader as ml

# Get the train and test data
#Checks if feature text file exists,
#if not creates file from wav

if(ml.is_loaded() == False):
    ml.load_wav()
   
train, test = fl.getFeatures()
print(train.shape)
    


# Run k-means
cl.kmeans(train, test)

print("Done")
