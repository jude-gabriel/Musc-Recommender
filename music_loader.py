import librosa
import numpy as np


files = librosa.util.find_files('./songs')
files = np.asarray(files)
features = np.array([])

# Extract the features
for file in files:
    y, sr = librosa.load(file)

    # Most songs are under 8 mins (10922034) and over 1 min (2324931)
    # Most songs fall around 3.9 min range so lets pad to there (5155826)
    # pad y to be length (will lengthen short songs and shorten long songs)
    y_pad = librosa.util.fix_length(y, size=sr*250)

    # Use padded song to get the features
    y_harmonic, y_percussive = librosa.effects.hpss(y_pad)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    rms = librosa.feature.rms(y_pad)
    melspect = librosa.feature.melspectrogram(y_pad)
    # tempogram = librosa.feature.tempogram(y_pad)
    # fourier = librosa.feature.fourier_tempogram(y_pad)
    # fourier = fourier.flatten()
    # spectral_centroid = librosa.feature.spectral_centroid(y_pad)
    # spectral_centroid = spectral_centroid.flatten()
    # chromagram = librosa.feature.chroma_cqt(y_pad)
    # chromagram = chromagram.flatten()

    # Feature vector: tempo, rms, melspectrum, tempogram, fourier, spectral_centroid, chromagram
    # This is a long feature vector. Maybe we could cut down length
    feature_vec = np.append(tempo, rms)
    feature_vec = np.append(feature_vec, melspect)
    # feature_vec = np.append(feature_vec, tempogram)
    # feature_vec = np.append(feature_vec, fourier)
    # feature_vec = np.append(feature_vec, spectral_centroid)
    # feature_vec = np.append(feature_vec, chromagram)

    # We need file name in feature vector, so we can check what song it is later
    feature_vec = np.append(feature_vec, str(file))

    # Append to the features list as a row
    features = np.append(features, feature_vec, axis=0)
features = features.reshape(files.size, int(len(features) / files.size))
with open('features.txt', 'a+') as outfile:
    np.savetxt(outfile, features, delimiter=',', fmt='%s')













#
# # Create the feature array
# features = features.reshape(files.size, 2)
# print(features.shape)

