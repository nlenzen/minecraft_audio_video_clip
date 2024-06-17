import os
import sys

sys.path.append(os.getcwd())

from minecraft_audio_video_clip.source.dataloading.datasets import VideoDataset
import torch
from minecraft_audio_video_clip import load_model
import torch
import numpy as np

# Paths
path_sand = 'videos/resampled/sand_resampled.mp4'
path_wood = 'videos/resampled/wood_resampled.mp4'
path_cobblestone = 'videos/resampled/cobblestone_1_resampled.mp4'
path_dirt = 'videos/resampled/dirt_resampled.mp4'
path_leaves = 'videos/resampled/leaves_1_resampled.mp4'
path_seeds = 'videos/resampled/seeds_resampled.mp4'
path_swim = 'videos/resampled/water_resampled.mp4'
path_underwater = 'videos/resampled/water_resampled.mp4'
path_place_sand = 'videos/resampled/place_sand_resampled.mp4'
path_place_wood = 'videos/resampled/place_wood_resampled.mp4'
path_place_dirt = 'videos/resampled/place_dirt_resampled.mp4'
path_place_cobblestone = 'videos/resampled/cobblestone_1_resampled.mp4'

model_path = 'checkpoints/model/10_layers_extended_5.pth'
model_cfg = 'minecraft_audio_video_clip/configs/10_layers.yaml'

savepath = 'datasets/embeddings/visual_embeddings.npz'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model(model_path, model_cfg, device)
model.eval()

test = VideoDataset(path_leaves, 32, 16)[11].unsqueeze(0)

leaves = VideoDataset(path_leaves, 32, 16)[11]
cobblestone = VideoDataset(path_cobblestone, 32, 16)[437]
dirt = VideoDataset(path_dirt, 32, 16)[41]
sand = VideoDataset(path_sand, 32, 16)[279]
seeds = VideoDataset(path_seeds, 32, 16)[57]
wood = VideoDataset(path_wood, 32, 16)[86]

batch1 = torch.stack([leaves, cobblestone, dirt, sand, seeds, wood])

swim = VideoDataset(path_swim, 32, 16)[11]
underwater = VideoDataset(path_underwater, 32, 16)[461]
place_cobblestone = VideoDataset(path_place_cobblestone, 32, 16)[355]
place_dirt = VideoDataset(path_place_dirt, 32, 16)[7]
place_sand = VideoDataset(path_place_sand, 32, 16)[53]
place_wood = VideoDataset(path_place_wood, 32, 16)[23]

batch2 = torch.stack([swim, underwater, place_cobblestone, place_dirt, place_sand, place_wood])

with torch.no_grad():
    batch1 = batch1.to(device)
    embedding1 = model.encode_video(batch1).cpu()
    del batch1

    batch2 = batch2.to(device)
    embedding2 = model.encode_video(batch2).cpu()
    del batch2

# Datasets
embeddings = {
    'destroy_leaves': embedding1[0],
    'get_cobblestone': embedding1[1],
    'get_dirt': embedding1[2],
    'get_sand': embedding1[3],
    'get_seeds': embedding1[4],
    'get_wood': embedding1[5],
    'go_swim': embedding2[0],
    'go_underwater': embedding2[1],
    'place_cobblestone': embedding2[2],
    'place_dirt': embedding2[3],
    'place_sand': embedding2[4],
    'place_wood': embedding2[5]
}

np.savez(savepath, **embeddings)
