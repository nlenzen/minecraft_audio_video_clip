# Imports
import os
from pydub import AudioSegment
import torch
import torchaudio
import librosa as lb
import math
import soundfile as sf
from .dataclass import DataWrapper

"""
Converts .ogg file format to .wav file format.

root_dir: directory where .ogg files are stored
target_dir: directory where .wav will be saved
"""
def audio_converter(root_dir, target_dir):
    for root, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith('.ogg'):
                path = os.path.join(root, name)
                rel_path = os.path.relpath(path, os.Path(root_dir))
                target_path = os.path.splitext(rel_path)[0] + '.wav'
                target_path = os.path.join(target_dir, target_path)
                directory_path = os.path.dirname(target_path)
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                audio = AudioSegment.from_file(path, format="ogg")
                audio.export(target_path, format="wav")


"""
extracts samples from given sound data and stores it in the data wrapper

name: base name of the samples. Each sample appends an index number to create a unique name
data: samples of the wavefile
sr: sample rate of the wavefile
sample_length: float depicting the wanted length of data samples. Example: 1 means a length of 1 second per sample, 
        1.5 means a length of 1.5 seconds per sample
data_struct: DataWrapper entity used to store the data
fill: accepts values in [None, 'zeros', 'avg', 'rand'] if not None it depicts the way, sampels with few datapoints are filled
dropout: value between 0 and 1 indicating the length the last sample needs to be considered
"""
def extract_samples(name, data, label, sample_length, data_struct, fill, dropout):
    sr = data_struct.sr
    pos = 0
    index = 1
    length = int(sr * sample_length)
    while pos + length - 1 < len(data):
        end = pos + length
        data_struct.add_data(name + '_' + str(index), data[pos:end], label)
        index += 1
        pos += length
    missing = len(data[pos:-1])
    if missing / length > dropout and fill is not None:
        if fill == 'zeros':
            sample = torch.zeros(length)
            sample[:missing] = data[pos:-1]
        if fill == 'rand':
            sample = torch.randn(length)
            sample[:missing] = data[pos:-1]
        if fill == 'avg':
            avg = torch.mean(data[pos:-1])
            sample = torch.ones(length) * avg
            sample[:missing] = data[pos:-1]
        data_struct.add_data(name + '_' + str(index), sample, label)



"""
Loads audio data from wave files in root_dir and its subdirectories. Audio samples are split into multiple subsamples
as indicated by scaling. If the last subsample of a sample has a length lower than indicated by dropout, it is not further 
used.
The function creates a label for each audio sample. The label is either the relative path from root_dir to the parent directory
of the audio file if the audio file is located in a subdirectory of root_dir or the name of the audio file if the file 
is located in root_dir.

root_dir: directory containing wave files
sample_length: float depicting the wanted length of data samples. Example: 1 means a length of 1 second per sample, 
        1.5 means a length of 1.5 seconds per sample
fill: accepts values in [None, 'zeros', 'avg', 'rand'] if not None it depicts the way, sampels with few datapoints are filled
dropout: value between 0 and 1 indicating the length the last sample needs to be considered
data_struct: DataWrapper entity to save sample names and data. If None a new entity will be created
"""
def load_audio(root_dir, sample_length=1, fill='avg', dropout=0, data_struct=None):
    if data_struct is None:
        data_struct = DataWrapper()
    index = 0
    for root, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith('.wav'):
                print("processing file {}".format(name))
                path = os.path.join(root, name)
                data, sr = torchaudio.load(path)
                if data_struct.sr is None:
                    data_struct.set_sr(sr)
                data = torch.mean(data, dim=0)  # Convert waveform to mono channel
                if sr != data_struct.sr:
                    data = lb.resample(data.numpy(), orig_sr=sr, target_sr=data_struct.sr)
                    data = torch.from_numpy(data)
                n = os.path.splitext(path)[0]
                n = os.path.relpath(n, root_dir)
                n = n.replace('/', '-')
                label = os.path.relpath(root, root_dir)
                if label == '.':
                    label = n
                extract_samples(n, data, label, sample_length, data_struct, fill, dropout)
    print("Done")

    return data_struct


"""
Saves each sample from the data wrapper into a wavefile. The name of the wavefile is given by the sample name plus 
the .wav file extension.

root_dir: root of the directory where the samples are supposed to be saved
data_struct: DataWrapper containing the sample data
"""
def save_audio(root_dir, data_struct):
    for name, sample, _ in data_struct:
        path = name + '.wav'
        path = os.path.join(root_dir, path)
        directory_path = os.path.dirname(path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        sample = sample.unsqueeze(0)
        torchaudio.save(path, sample, data_struct.sr)


"""
Extracts audio from all videos in root_dir and its subdirectories and saves it as a .wav file in target_dir.
Directory structure of subdirectories is preserved.

root_dir: Root of the directory containing mp4 files
target_dir: Root of directory where the resulting wav files are stored
sr: Sample rate of the resulting wav files
"""
def extract_audio(root_dir, target_dir, sr):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp4'):
                print('Processing file: {}'.format(file))
                path = os.path.join(root, file)
                relative_path = os.path.relpath(path, root_dir)
                target_path = os.path.join(target_dir, relative_path)
                target_path = os.path.splitext(target_path)[0]
                if not os.path.exists(os.path.dirname(target_path)):
                    os.makedirs(os.path.dirname(target_path))
                command = 'ffmpeg -i {} -acodec pcm_s16le -ar {} {}.wav'.format(path, sr, target_path)
                # Check if source is run in ipynp notebook. Command needs to start with '!' to work in google colab
                if hasattr(__builtins__, '__IPYTHON__'):
                    command = '!' + command
                os.system(command)


"""
Splits audio into multiple audio files. Can be used if an audio file is too large to be loaded in one piece.
New files are stored in the same directory

target_path: Path to large audio file
length: Duration of the generated audio files. In seconds
"""
def split_audio(target_path, duration):
    dir_path = os.path.dirname(target_path)

    start = 0
    index = 0
    data, sr = lb.load(target_path, offset=start, duration=duration)
    while math.isclose(len(data)/sr, duration, abs_tol=0.00003):
        filename = str(index) + '_' + os.path.basename(target_path)
        path = os.path.join(dir_path, filename)
        sf.write(path, data, sr)
        index += 1
        start += duration
        data, sr = lb.load(target_path, offset=start, duration=duration)
    filename = str(index) + '_' + os.path.basename(target_path)
    path = os.path.join(dir_path, filename)
    sf.write(path, data, sr)


"""
Resamples all wav files in given directory and its subdirectories to the provided sample rate.

root_dir: Root directory
sample_rate: target sample rate
"""
def resample_files(root_dir, sample_rate):
    for root, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith('.wav'):
                path = os.path.join(root, name)
                data, sr = lb.load(path)
                data = lb.resample(data, orig_sr=sr, target_sr=sample_rate)
                sf.write(path, data, sample_rate)
