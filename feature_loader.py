import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



def getFeatures(filename):
    # Load in features and split into data and labels
    features = np.loadtxt(filename, delimiter=",", dtype='str')
    feature_vecs = features[:, 0:len(features[0]) - 1].astype(float)
    feature_names = features[:, len(features[0]) - 1]

    # Run PCA to capture 99% of variance
    pca_model = PCA(n_components=0.9999)
    pca_vals = pca_model.fit_transform(feature_vecs)
    features = np.concatenate((pca_vals, feature_names.reshape(-1, 1)), axis=1)

    # printing explaination of variance
    print('variance: ', pca_model.explained_variance_)
    print('variance ratio: ', pca_model.explained_variance_ratio_)
    print('cumulative sum of variance: ', pca_model.explained_variance_.cumsum())

    # Generate a train/test split
    # Train = users listening history
    # Test = potential songs to recommend
    train, test = train_test_split(features, test_size=0.2)
    return train, test
