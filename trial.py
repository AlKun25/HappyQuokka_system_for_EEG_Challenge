import numpy as np

path = "/home/kunal/eeg_data/derivatives/downsample/train/train_-_sub-085_-_podcast_37_-_mel_-_70.npy"
mel = np.load(path)
print(mel.shape)