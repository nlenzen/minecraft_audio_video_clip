# Imports
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.utils.data import DataLoader
from minecraft_audio_video_clip.source.dataloading.datasets import WaveDataset, VideoDataset
from minecraft_audio_video_clip.source.load import load_model


def save(audio, path):
    audio_embeds = np.concatenate(audio, axis=0)

    audio_path = os.path.join(path, 'audio_embeddings')
    if os.path.exists(audio_path):
        os.remove(audio_path)
    np.savez(audio_path, audio=audio_embeds)


# Paths
model_path = 'checkpoints/model_mineclip/trained_no_act.pth'
model_cfg = 'minecraft_audio_video_clip/configs/mineclip/10_layers_mineclip_no_activation.yaml'
video_path = 'videos/resampled/minecraft_video_1_16000hz.mp4'
audio_path = 'datasets/from_videos/subfile'
savepath = 'datasets/from_videos/encoded_subfiles'

# Params
batch_size = 16
num_workers = 2
prefetch_factor = 2
shuffle = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, model_cfg, device)
wav_dataset = WaveDataset(audio_path)
# video_dataset = VideoDataset(video_path, fps, num_frames)
wav_dataloader = DataLoader(wav_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=prefetch_factor)

embeddings_audio = []
with torch.no_grad():
    for i, batch in enumerate(wav_dataloader):
        batch = batch.to(device)
        audio_encs = model.encode_audio(batch)
        embeddings_audio.append(audio_encs.cpu().numpy())

        if i % 100 == 0:
            save(embeddings_audio, savepath)

    save(embeddings_audio, savepath)

