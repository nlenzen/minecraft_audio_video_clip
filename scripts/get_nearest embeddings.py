# Imports
import os
import sys
import hashlib

sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.utils.data import DataLoader
from minecraft_audio_video_clip.source.dataloading.datasets import WaveDataset, VideoDataset
from minecraft_audio_video_clip.source.load import load_model
from mineclip import MineCLIP


MINECLIP_CONFIG = {
    'arch': "vit_base_p16_fz.v2.t2",
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': "attn.d2.nh8.glusw",
    'resolution': [160, 256],
    'ckpt': {
        'path': "data/weights/mineclip/attn.pth",
        'checksum': 'b5ece9198337cfd117a3bfbd921e56da'
    }
}


def load(cfg, device):
    cfg = cfg.copy()
    ckpt = cfg.pop("ckpt")
    assert hashlib.md5(open(ckpt['path'], "rb").read()).hexdigest() == ckpt['checksum'], "broken ckpt"

    model = MineCLIP(**cfg).to(device)
    model.load_ckpt(ckpt['path'], strict=True)
    return model


# Paths
model_path = 'checkpoints/model_mineclip/no_act_L1_extended_dataset.pth'
model_cfg = 'minecraft_audio_video_clip/config/mineclip/10_layers_mineclip_no_activation.yaml'
video_path = 'videos/resampled/minecraft_video_1_16000hz.mp4'
audio_path = 'datasets/from_videos/encoded_subfiles/train_audio_embeddings.npz'
savepath = 'datasets/from_videos/encoded_subfiles'

# Params
fps = 32
num_frames = 16
batch_size = 32
num_workers = 2
prefetch_factor = 2
shuffle = False
index = 1
k = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, model_cfg, device)
video_dataset = VideoDataset(video_path, fps, num_frames)
data = np.load(audio_path)
audio_embeddings = data['audio']

video_clip = video_dataset[index]
video_clip = video_clip.unsqueeze(0).to(device)

with torch.no_grad():
    video_embedding = model.encode_video(video_clip)
    video_embedding = model.project_video_embeddings(video_embedding)

audio_embeddings = torch.from_numpy(audio_embeddings).to(device)
probs = (video_embedding @ audio_embeddings.T).softmax(dim=-1).cpu()
top_k = probs.topk(5)
indices = [i for i, value in enumerate(probs) if value in top_k]
print(indices)
