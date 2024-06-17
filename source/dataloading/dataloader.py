# Imports
import os
import cv2
import csv
import math
import numpy as np
from moviepy.editor import *
import random
import decord as de


class EmbeddingLoader:
    def __init__(self,
                 batch_size,
                 train_data_path,
                 test_data_path,
                 train_val_ratio=(8, 2),
                 overlap=0.75,
                 sample_length=1,
                 n_frames=16,
                 sr=16000,
                 fps=32):
        self.train_video_data = []
        self.train_audio_data = []
        self.test_video_data = []
        self.test_audio_data = []
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.overlap = overlap
        self.sample_length = sample_length
        self.n_frames = n_frames
        self.sr = sr
        self.fps = fps

        if os.path.exists(train_data_path):
            print("Loading training data...")
            data = np.load(train_data_path)
            self.train_video_data = data['video']
            self.train_audio_data = data['audio']

        if os.path.exists(test_data_path):
            print("Loading test data...")
            data = np.load(test_data_path)
            self.test_video_data = data['video']
            self.test_audio_data = data['audio']

    def randomize_sample_order(self):
        p = np.random.permutation(len(self.train_audio_data))
        self.train_video_data = self.train_video_data[p]
        self.train_audio_data = self.train_audio_data[p]

    def len_train_batches(self):
        return math.ceil(len(self.train_video_data) / float(self.batch_size))

    def len_test_batches(self):
        return math.ceil(len(self.test_video_data) / float(self.batch_size))

    def len_test_set(self):
        return len(self.test_video_data)

    def len_train_set(self):
        return len(self.train_video_data)

    # Draws random ramples from the test set. The number of samples is defined by self.batch_size
    def get_random_test_samples(self, size):
        p = np.random.choice(len(self.test_video_data), size)
        video_embeddings = self.test_video_data[p]
        audio_embeddings = self.test_audio_data[p]

        return video_embeddings, audio_embeddings

    def get_random_train_samples(self, size):
        p = np.random.choice(len(self.test_video_data), size)
        video_embeddings = self.train_video_data[p]
        audio_embeddings = self.train_audio_data[p]

        return video_embeddings, audio_embeddings

    def get_batch(self, batch_num):
        start = batch_num * self.batch_size
        end = (batch_num + 1) * self.batch_size
        end = end if end <= len(self.train_video_data) else len(self.train_video_data)

        video_embeddings = self.train_video_data[start: end]
        audio_embeddings = self.train_audio_data[start: end]

        return video_embeddings, audio_embeddings

    def __len__(self):
        return self.len_train_batches()

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.len_train_batches():
            raise StopIteration
        video_embeds, audio_embeds = self.get_batch(self.index)
        self.index += 1
        return video_embeds, audio_embeds

# ======================================================================================================================
# Loads video sources completely into RAM.
# Slower at startup but the data should be loaded much quicker at train time.
# ======================================================================================================================


class Dataloader:
    def __init__(self,
                 batch_size,
                 train_data_path=None,
                 test_data_path=None,
                 train_val_ratio=(8, 2),
                 overlap=0.75,
                 sample_length=1,
                 n_frames=16,
                 sr=16000,
                 fps=32):
        self.train_data = []
        self.test_data = []
        self.source_dict = {}
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.overlap = overlap
        self.sample_length = sample_length
        self.n_frames = n_frames
        self.sr = sr
        self.fps = fps

        ctx = de.cpu(0)
        if train_data_path is not None:
            self.train_data_path = train_data_path
            if os.path.exists(train_data_path):
                with open(self.train_data_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row[0] not in self.source_dict:
                            self.source_dict[row[0]] = de.AVReader(row[0], ctx, sample_rate=self.sr, mono=True)
                        self.train_data.append(row)
                    f.close()
        else:
            self.train_data_path = 'train_data.csv'

        if test_data_path is not None:
            self.test_data_path = test_data_path
            if os.path.exists(test_data_path):
                with open(self.test_data_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row[0] not in self.source_dict:
                            self.source_dict[row[0]] = de.AVReader(row[0], ctx, sample_rate=self.sr, mono=True)
                        self.test_data.append(row)
                    f.close()
        else:
            self.test_data_path = 'test_data.csv'

    def create_data(self, video_path):
        ctx = de.cpu()
        train_samples = []
        test_samples = []

        reader = de.AVReader(video_path, ctx, sample_rate=self.sr, mono=True)
        video_length = len(reader)
        self.source_dict[video_path] = reader
        slide = self.fps - self.overlap * self.fps
        slide = int(slide * self.sample_length)
        idx = 0
        start = 0
        end = int(self.sample_length * self.fps)
        while end < video_length:
            sample = [video_path, start, end]
            if idx in range(0, self.train_val_ratio[0]):
                train_samples.append(sample)
                idx += 1
            if idx in range(self.train_val_ratio[0], self.train_val_ratio[0] + self.train_val_ratio[1]):
                test_samples.append(sample)
                idx += 1
            if idx >= (self.train_val_ratio[0] + self.train_val_ratio[1]) - 1:
                idx = 0
            start += slide
            end += slide

        self.test_data.extend(test_samples)
        self.train_data.extend(train_samples)
        with open(self.train_data_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(train_samples)
            f.close()
        with open(self.test_data_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(test_samples)
            f.close()

    def load_train_set(self, path, add=True):
        new_data = []
        ctx = de.cpu(0)
        if not add:
            self.source_dict = {}
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in self.source_dict:
                    self.source_dict[row[0]] = de.AVReader(row[0], ctx, sample_rate=self.sr, mono=True)
                new_data.append(row)
            f.close()
        if add:
            self.train_data.extend(new_data)
        else:
            self.train_data = new_data

    def load_test_set(self, path, add=True):
        new_data = []
        ctx = de.cpu(0)
        if not add:
            self.source_dict = {}
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in self.source_dict:
                    self.source_dict[row[0]] = de.AVReader(row[0], ctx, sample_rate=self.sr, mono=True)
                new_data.append(row)
            f.close()
        if add:
            self.test_data.extend(new_data)
        else:
            self.test_data = new_data

    def get_batch(self, batch_num):
        start_idx = batch_num * self.batch_size
        end_idx = (batch_num + 1) * self.batch_size
        if start_idx > len(self.train_data):
            return None
        if end_idx > len(self.train_data):
            return self.train_data[start_idx:]
        return self.train_data[start_idx:end_idx]

    def get_test_batch(self, batch_num):
        start_idx = batch_num * self.batch_size
        end_idx = (batch_num + 1) * self.batch_size
        if start_idx > len(self.test_data):
            return None
        if end_idx > len(self.test_data):
            return self.test_data[start_idx:]
        return self.test_data[start_idx:end_idx]

    def load_data(self, batch):
        if isinstance(batch, int):
            batch = self.get_batch(batch)
        batch = np.array(batch)
        if batch is None:
            return
        frame_data = []
        audio_data = []

        for sample in batch:
            reader = self.source_dict[sample[0]]
            start = int(sample[1])
            end = int(sample[2])
            audio, video = reader[start:end]
            video = video.asnumpy()
            if video.shape[0] == 1:
                frames = np.squeeze(video, axis=0)
            else:
                spacing = np.linspace(0, video.shape[0] - 1, num=self.n_frames, dtype=int)
                frames = np.take(video, spacing, axis=0)
            buffer = np.array([])
            for a in audio:
                buffer = np.append(buffer, a.asnumpy())

            frame_data.append(frames)
            audio_data.append(buffer)

        return np.array(frame_data), np.array(audio_data)

        """
        for source in self.source_dict.items():
            name = source[0]
            reader = source[1]
            data = [entry for entry in batch if entry[0] == name]
            for d in data:
                start = int(d[1] * self.fps)
                end = int(d[2] * self.fps)
                audio, video = reader[start: end]
                spacing = np.linspace(0, len(video) - 1, num=self.n_frames, dtype=int)
                frames = [f for i, f in enumerate(video) if i in spacing]
                audio = list(chain.from_iterable(audio))

                frame_data.append(frames)
                audio_data.append(audio)
        return np.array(frame_data), np.array(audio_data)
        """

    def load_all_data(self, source):
        if source == 'train':
            return self.load_data(self.train_data)
        if source == 'test':
            return self.load_data(self.test_data)
        else:
            return None, None

    def randomize_sample_order(self):
        random.shuffle(self.train_data)

    def len_batches(self):
        return math.ceil(len(self.train_data) / float(self.batch_size))

    def len_test_batches(self):
        return math.ceil(len(self.test_data) / float(self.batch_size))

    def __len__(self):
        return self.len_batches()

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.len_batches():
            raise StopIteration
        batch = self.get_batch(self.index)
        self.index += 1
        return batch

