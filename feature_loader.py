import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

# run a PCA and return the model



def runPca(data, n_components):
    pca = PCA(n_components=n_components)
    principleComponents = pca.fit_transform(data)
    print(principleComponents)
    #time.sleep(100)

    pca_model = decomposition.PCA(n_components=n_components)  # make PCA model with n number of components
    variance = pca_model.fit(data)  # setting variance to an object

    # printing explaination of variance
    print('variance: ', pca_model.explained_variance_)
    print('variance ratio: ', pca_model.explained_variance_ratio_)
    print('cumulative sum of variance: ', pca_model.explained_variance_.cumsum())

    pca_data = pca_model.transform(data) # transforming model

    return  pca_data

# computes pca and writes to file, returns train and test split datasets
def writeToFile(pca_data, fileNames):
    fileWrite = open("featuresAfterPCA.txt", "w")
    count = 0
    for i in pca_data:
        for a in i:
            fileWrite.write(str(a) + ', ')
        fileWrite.write(fileNames[count] + '\n')
        count = count + 1

# get features
def getFeatures(n_components):
    features = np.loadtxt("features.txt", delimiter=",", dtype='float', comments='G',usecols=[0,497])
    fileNames = np.loadtxt("features.txt", delimiter=",",dtype='str', comments='00000000000', usecols=[498])

    pca = runPca(features, n_components)
    writeToFile(pca, fileNames) # runs and writes the PCA to file 'featuresAfterPCA.txt'

    train_music, test_music = train_test_split(features, test_size=1/7.0, random_state=0)


    return(train_music, test_music)

    # standardizing the datas features
    scaler = StandardScaler()
    scaler.fit(train_music)
    train_music = scaler.transform(train_music)
    test_music = scaler.transform(test_music)

    # return(train_music, test_music) # return for post standard scale train and test sets


# function call, runs pca on dataset in features.txt file, n_compenents size
n_components=2
getFeatures(n_components)