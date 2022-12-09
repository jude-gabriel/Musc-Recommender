import music_loader as ml
import feature_loader as fl
import clustering as cl
import genre_analysis as ga
import minDistance as md

# Create the feature files
hasFeatures = input("Are feature files made? 1 for yes, 0 for no: ")
if hasFeatures == '0':
    ml.music_loader()

# For each set of features do clustering to get recommended songs and then find sub genres within clusters
features = ['features1.txt', 'features2.txt', 'features3.txt', 'features4.txt']
count = 1
for feature in features:
    print("\n\nFeature set " + str(count))
    count = count + 1

    # Get the train and test data
    train, test = fl.getFeatures(feature)

    # Run k-means to get list of songs to recommend
    initial_clusters, initial_labels, final_clusters, final_labels, most_dense = cl.kmeans(train, test)

    # Do genre analysis to determine before and after sub-genres
    ga.analysis(initial_clusters, initial_labels, final_clusters, final_labels)

    # Run k-means and use closest based on distance to recommend 5 songs
    md.getMindDistances(train, test, most_dense)