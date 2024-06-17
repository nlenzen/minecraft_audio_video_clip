import os
import sys

sys.path.append(os.getcwd())

import math
import torch
import numpy as np
from minecraft_audio_video_clip.source.load import load_model

# Paths
model_path = 'checkpoints/evaluation_checkpoints/clip_projection_bn_no_activation.pth'
model_cfg = 'minecraft_audio_video_clip/configs/projection_bn_no_activation.yaml'
embeddings_path = 'datasets/embeddings/extended/train_extended.npz'
savepath = 'datasets/audio_embeddings/train_audio_embeddings_projection.npz'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model(model_path, model_cfg, device)
data = np.load(embeddings_path)
video_data = data['video']
audio_data = data['audio']

batch_size = 2048
batch_num = math.ceil(len(audio_data) / batch_size)


embeddings = []
with torch.no_grad():
    for i in range(batch_num):
        start = i * batch_size
        end = start + batch_size
        end = end if end < len(audio_data) else len(audio_data)

        batch = audio_data[start:end]
        batch = torch.from_numpy(batch).to(device)
        audio_embeddings = model.project_audio_embeddings(batch).cpu()
        embeddings.append(audio_embeddings.numpy())

embeddings = np.concatenate(embeddings, axis=0)
np.savez(savepath, audio=embeddings, video=video_data)
