import os
import sys
sys.path.append(os.getcwd())
import torch
from torchaudio import load
from minecraft_audio_video_clip.source import load_model
from minecraft_audio_video_clip.source.preprocess import make_features
import time

model_path = 'checkpoints/model/10_layers_extended.pth'
model_cfg = 'minecraft_audio_video_clip/configs/10_layers.yaml'
wavefile = 'datasets/test_files/dirt/dirt_1.wav'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(model_path, model_cfg, device)
audio, sr = load(wavefile)


images = torch.randn(1, 16, 160, 256, 3).to(device)

start_time = time.time()
with torch.no_grad():
    vid_embed = model.forward_video(images)
    video_time = time.time()

    features = make_features(audio, sr).to(device)
    predictions = model.forward_audio(features.unsqueeze(0))
    audio_time = time.time()

print(f'Time taken to encode video: {video_time - start_time}s')
print(f'Time taken to encode audio: {audio_time - video_time}s')
