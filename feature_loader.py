import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def getFeatures():
    features = np.loadtxt("features.txt", delimiter=",", dtype='str')
    feature_vecs = features[:, 0:len(features[0]) - 1].astype(float)
    feature_names = features[:, len(features[0]) - 1]

    pca_model = PCA(n_components=0.9)
    pca_vals = pca_model.fit_transform(feature_vecs)
    features = np.concatenate((pca_vals, feature_names.reshape(-1, 1)), axis=1)

    # printing explaination of variance
    print('variance: ', pca_model.explained_variance_)
    print('variance ratio: ', pca_model.explained_variance_ratio_)
    print('cumulative sum of variance: ', pca_model.explained_variance_.cumsum())

    train, test = train_test_split(features, test_size=0.2)

    return train, test
