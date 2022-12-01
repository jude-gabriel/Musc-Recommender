import librosa
import numpy as np
import os

def load_wav():
    files = librosa.util.find_files('./songs')
    files = np.asarray(files)
    features = np.array([])

    # Extract the features
    for file in files:
        print(file)
        y, sr = librosa.load(file)

        # calls function to extract features, 2 feature function exist currently
        feature_vec = get_features1(y, sr, file)

        # Append to the features list as a row
        features = np.append(features, feature_vec, axis=0)

    features = features.reshape(files.size, int(len(features) / files.size))

    # to reduce textfile length, all feature vectors become 32bit floats
    features[:, 0:len(features[0]) - 1] = features[:, 0:len(features[0]) - 1].astype(np.float32)
    with open('features.txt', 'a+') as outfile:
        np.savetxt(outfile, features, delimiter=',', fmt='%s')


# get features 1
def get_features1(y, sr, file):
    # pad y to be length (will lengthen short songs and shorten long songs)
    y_pad = librosa.util.fix_length(y, size=sr * 250)

    # Mel Frequency Cepstral Coefficients (MFCC)
    mfcc = librosa.feature.mfcc(y=y_pad, sr=sr)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    mfcc_vec = np.concatenate((mfcc_mean, mfcc_min, mfcc_max))

    # Mel Spectrogram
    melspect = librosa.feature.melspectrogram(y=y_pad)
    melspect_mean = melspect.mean(axis=1)
    melspect_min = melspect.min(axis=1)
    melspect_max = melspect.max(axis=1)
    melspect_vec = np.concatenate((melspect_mean, melspect_min, melspect_max))

    # Chroma vector
    chroma = librosa.feature.chroma_stft(y=y_pad, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_min = chroma.min(axis=1)
    chroma_max = chroma.max(axis=1)
    chroma_vec = np.concatenate((chroma_mean, chroma_min, chroma_max))

    # Tonal centorid features (Tonnetz)
    tonnetz = librosa.feature.tonnetz(y=y_pad, sr=sr)
    tonnetz_mean = tonnetz.mean(axis=1)
    tonnetz_min = tonnetz.min(axis=1)
    tonnetz_max = tonnetz.max(axis=1)
    tonnetz_vec = np.concatenate((tonnetz_mean, tonnetz_min, tonnetz_max))

    feature_vec = np.append(mfcc_vec, melspect_vec)
    feature_vec = np.append(feature_vec, chroma_vec)
    feature_vec = np.append(feature_vec, tonnetz_vec)

    # We need file name in feature vector, so we can check what song it is later
    feature_vec = np.append(feature_vec, str(file))
    return feature_vec


load_wav()