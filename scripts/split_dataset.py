import numpy as np
import os
import sys

sys.path.append(os.getcwd())

path = 'datasets/generation/6200_iterations.npz'
savepath = 'datasets/generation/test_train_datasets'

data = np.load(path)
img_data = data['video']
audio_data = data['audio']

train_images = []
test_images = []
train_audio = []
test_audio = []

idx = 0
for i in range(img_data.shape[0]):
    if idx in range(0, 8):
        train_images.append(img_data[i])
        train_audio.append(audio_data[i])
        idx += 1
    if idx in range(8, 10):
        test_images.append(img_data[i])
        test_audio.append(audio_data[i])
        idx += 1
    if idx >= 9:
        idx = 0

np.savez(os.path.join(savepath, 'test_embeddings'), audio=test_audio, video=test_images)
np.savez(os.path.join(savepath, 'train_embeddings'), audio=train_audio, video=train_images)


