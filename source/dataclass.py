# imports
import librosa as lb
import torch


"""
Simple wrapper class to store needed data

sr: Sample rate od the samples
"""
class DataWrapper:
    def __init__(self, sr=None):
        self.names = []
        self.samples = torch.tensor([])
        self.labels = []
        self.sr = sr

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.names):
            raise StopIteration
        name = self.names[self.index]
        sample = self.samples[self.index]
        label = self.labels[self.index]
        self.index += 1
        return name, sample, label

    def __len__(self):
        return len(self.names)

    def get_sample(self, pos):
        return self.names[pos], self.samples[pos], self.labels[pos]

    def add_data(self, name, sample, label):
        self.names.append(name)
        self.samples = torch.cat((self.samples, sample.unsqueeze(0)), dim=0)
        self.labels.append(label)

    def set_sr(self, sr):
        self.sr = sr

    def resample(self, sr):
        resampled = lb.resample(self.samples.numpy(), orig_sr=self.sr, target_sr=sr, axis=1)
        new_wrapper = DataWrapper(sr)
        new_wrapper.names = self.names.copy()
        new_wrapper.samples = resampled
        new_wrapper.labels = self.labels.copy()
        new_wrapper.to_tensor()

        return new_wrapper

    def resample_self(self, sr):
        self.samples = lb.resample(self.samples.numpy(), orig_sr=self.sr, target_sr=sr, axis=1)
        self.sr = sr
        self.to_tensor()

    def to_tensor(self):
        self.samples = torch.from_numpy(self.samples)

    def to_nparray(self):
        self.samples = self.samples.numpy()


