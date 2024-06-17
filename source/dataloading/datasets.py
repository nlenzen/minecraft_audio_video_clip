# Imports
import os
import numpy as np
import torch
from torchaudio import load
from torch.utils.data import Dataset
from minecraft_audio_video_clip.source.preprocess import make_features
import cv2


class EmbeddingsDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.audio = data['audio']
        self.video = data['video']

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        audio = torch.from_numpy(self.audio[idx])
        video = torch.from_numpy(self.video[idx])

        return audio, video


class WaveDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        length = len([name for name in os.listdir(self.root_dir) if name.endswith('.wav')])
        return length

    def __getitem__(self, idx):
        extension = '.wav'
        # index = str(idx).zfill(3)
        index = str(idx)
        filename = 'out' + index + extension
        path = os.path.join(self.root_dir, filename)
        data, sr = load(path)
        # data = data[:, :sr]
        if self.transform:
            data = self.transform(data)

        spectrogram = make_features(data, sr)

        return torch.tensor(spectrogram)


class AudioDataset(Dataset):
    def __init__(self, audio_path, normalize=True, return_spec=True):
        audio, self.sr = load(audio_path, normalize=normalize)
        self.audio = audio[0]
        self.return_spec = return_spec

    def __len__(self):
        return len(self.audio) // self.sr

    def __getitem__(self, idx):
        start = idx * self.sr
        end = (idx + 1) * self.sr
        data = self.audio[start:end]
        #assert data == self.audio.shape
        if self.return_spec:
            data = make_features(data.unsqueeze(0), self.sr)

        return torch.tensor(data)


class VideoDataset(Dataset):
    def __init__(self, video_path, fps, num_frames):
        self.video_path = video_path
        self.fps = fps
        self.num_frames = num_frames

    def __len__(self):
        cap = cv2.VideoCapture(self.video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return length // self.fps

    def __getitem__(self, idx):
        start_video = int(idx * self.fps)
        end_video = int(start_video + self.fps)
        spacing = np.linspace(start_video, end_video - 1, num=self.num_frames, dtype=int)
        frames = []
        video_cap = cv2.VideoCapture(self.video_path)
        for i in spacing:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.tensor(frame))
        frames = torch.stack(frames, dim=0)

        return frames


class AVDataset(Dataset):
    def __init__(self, video_path, audio_path, fps, num_frames):
        audio, sr = load(audio_path)
        self.audio = audio.squeeze(0)
        self.sr = sr
        self.video_cap = cv2.VideoCapture(video_path)
        self.fps = fps
        self.num_frames = num_frames

    def __len__(self):
        length = int(len(self.audio) / self.sr)
        length = (length - 1) * 4 + 1
        return length

    def __getitem__(self, idx):
        start_video = int(idx * 0.25 * self.fps)
        end_video = int(start_video + self.fps)
        spacing = np.linspace(start_video, end_video - 1, num=self.num_frames, dtype=int)
        frames = []
        for i in spacing:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.video_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.tensor(frame))
        frames = torch.stack(frames, dim=0)

        start_audio = int(idx * 0.25 * self.sr)
        end_audio = int(start_audio + self.sr)
        audio = self.audio[start_audio:end_audio]
        features = make_features(audio.unsqueeze(0), self.sr)

        return features, frames
