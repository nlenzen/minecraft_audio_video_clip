# Imports
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from minecraft_audio_video_clip.source import load_model
from minecraft_audio_video_clip.source.preprocess import make_features
from minecraft_audio_video_clip.source.dataloading.datasets import AVDataset
import decord as de
import math


def encode_and_save_embeddings(model, dataloader, savepath, number):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # chkpt = torch.load(chkpt_path, map_location=device)
    # model = load_model(chkpt, cfg_path, device)

    # Freeze encoder parameters - we do not want to train them
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    if model.temporal_encoder is not None:
        for param in model.temporal_encoder.parameters():
            param.requires_grad = False

    train_audio = []
    train_video = []
    test_audio = []
    test_video = []

    model.eval()

    with torch.no_grad():
        idx = 0
        num_batch = 0
        for audio_data, video_data in tqdm(dataloader, desc="Encoding and saving embeddings: "):
            video_data = video_data.to(device)
            audio_data = audio_data.to(device)
            # features = torch.stack([make_features(feature.unsqueeze(0), 16000) for feature in audio_data])
            # features = features.to(device)

            audio_encs = model.encode_audio(audio_data).cpu()
            video_encs = model.encode_video(video_data).cpu()
            for i in range(audio_encs.shape[0]):
                if idx in range(0, 8):
                    train_video.append(np.expand_dims(video_encs[i].numpy(), axis=0))
                    train_audio.append(np.expand_dims(audio_encs[i].numpy(), axis=0))
                    idx += 1
                if idx in range(8, 10):
                    test_video.append(np.expand_dims(video_encs[i].numpy(), axis=0))
                    test_audio.append(np.expand_dims(audio_encs[i].numpy(), axis=0))
                    idx += 1
                if idx >= 9:
                    idx = 0

            if num_batch % 100 == 0:
                save(train_audio, train_video, test_audio, test_video, savepath, number)
            num_batch += 1

    save(train_audio, train_video, test_audio, test_video, savepath, number)
    return train_audio, train_video, test_audio, test_video


def save(train_audio, train_video, test_audio, test_video, path, number):
    train_audio_embeds = np.concatenate(train_audio, axis=0)
    train_video_embeds = np.concatenate(train_video, axis=0)

    test_audio_embeds = np.concatenate(test_audio, axis=0)
    test_video_embeds = np.concatenate(test_video, axis=0)

    test_path = os.path.join(path, 'test_embeddings_video_{}'.format(number))
    train_path = os.path.join(path, 'train_embeddings_video_{}'.format(number))
    if os.path.exists(test_path):
        os.remove(test_path)
    if os.path.exists(train_path):
        os.remove(train_path)
    np.savez(train_path, audio=train_audio_embeds, video=train_video_embeds)
    np.savez(test_path, audio=test_audio_embeds, video=test_video_embeds)


conf_path = 'minecraft_audio_video_clip/configs/10_layers.yaml'
model_path = 'checkpoints/model/10_layers.pth'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Loading model')
model = load_model(model_path, conf_path, device)
print('Done')

print('Loading dataset')
video_paths = [('videos/resampled/cobblestone_1_resampled.mp4', 'videos/resampled/cobblestone_1_resampled.wav'),
               ('videos/resampled/cobblestone_2_resampled.mp4', 'videos/resampled/cobblestone_2_resampled.wav'),
               ('videos/resampled/place_wood_resampled.mp4', 'videos/resampled/place_wood_resampled.wav'),
               ('videos/resampled/place_dirt_resampled.mp4', 'videos/resampled/place_dirt_resampled.wav'),
               ('videos/resampled/place_sand_resampled.mp4', 'videos/resampled/place_sand_resampled.wav')]

ctx = de.cpu(0)
de.bridge.set_bridge('torch')
print('Done')


print('Creating embeddings')


# Params
path = 'datasets/embeddings/extended'
batch_size = 20
fps = 32
num_frames = 16
num_workers = 2
prefetch_factor = 1
num_threads = 2

offset = 15

for i, paths in enumerate(video_paths):
    print(f'Processing video {paths[0]}')
    video_path = paths[0]
    audio_path = paths[1]
    dataset = AVDataset(video_path, audio_path, fps, num_frames)
    print('Processing {} samples'.format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor)
    encode_and_save_embeddings(model, dataloader, path, i + offset)

print('Done')


