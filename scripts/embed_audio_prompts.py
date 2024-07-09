import sys
import os

sys.path.append(os.getcwd())

from minecraft_audio_video_clip.source import load_model
from minecraft_audio_video_clip.source.preprocess import make_features
from minecraft_audio_video_clip.source.cvae.cvae import load_cvae

import torch
import torchaudio
import numpy as np


def get_prior_embed(path, clip, prior, device):
    """Get the embed processed by the prior."""
    # Load audio file
    data, sr = torchaudio.load(path)
    features = make_features(data, sr).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        embed = clip.encode_audio(features).detach()
        audio_embed = clip.project_audio_embeddings(embed).detach().cpu().numpy()
    with torch.no_grad(), torch.cuda.amp.autocast():
        audio_prompt_embed = prior(torch.tensor(audio_embed).float().to(device)).cpu().detach().numpy()
    return audio_prompt_embed


# Paths
prior_path = 'checkpoints/cvae/audio_prior_projection.pth'
prior_cfg = 'minecraft_audio_video_clip/configs/cvae/cvae.yaml'
model_path = 'checkpoints/evaluation_checkpoints/clip_projection_bn_no_activation.pth'
model_cfg = 'minecraft_audio_video_clip/configs/audio_model_cfg.yaml'

embeddings_path = 'datasets/extended_audio_prompts'
savepath = 'datasets/embedded_prompts/audio_prompts_base_new.npz'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prior = load_cvae(prior_cfg, prior_path, device)
clip = load_model(model_path, model_cfg, device)

if os.path.isdir(embeddings_path):
    custom_prompt_embeds = {}
    entries = os.listdir(embeddings_path)
    dirs = [name for name in entries if os.path.isdir(os.path.join(embeddings_path, name))]
    for directory in dirs:
        dirname = os.path.join(embeddings_path, directory)
        files = os.listdir(dirname)
        files = sorted(files)
        for file in files:
            if file.endswith('.wav'):
                print("Processing file {} form directory {}".format(file, directory))
                name = os.path.join(dirname, file)
                prompt_embed = get_prior_embed(name, clip, prior, device)
                custom_prompt_embeds[directory] = prompt_embed

    np.savez(savepath, **custom_prompt_embeds)
