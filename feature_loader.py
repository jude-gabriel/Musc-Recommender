import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def getFeatures():
    features = np.loadtxt("features.txt", delimiter=",", dtype='str')


    # Todo: Run PCA on feature vectors. Need to find which features matter
    # Note: values stores as strings. Last values in the array are the names of the songs
    # When doing PCA make sure to only use the first n-1 columns and append the file names back after


    # Run PCA analysis and then ...
    train, test = train_test_split(features, test_size=0.2)
    return train, test