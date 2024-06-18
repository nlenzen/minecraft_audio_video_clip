import os
import sys

sys.path.append(os.getcwd())

import torch
from minecraft_audio_video_clip.source.load import load_model
from moviepy.editor import VideoFileClip, concatenate_videoclips
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

from minecraft_audio_video_clip.source.dataloading.datasets import VideoDataset, AudioDataset

# Plot Audio Video CLIP training metrics ===============================================================================


def get_epoch_means(metric, num_batches):
    result = []
    while len(metric) > 0:
        values = metric[:num_batches]
        metric = metric[num_batches:]
        result.append(np.mean(values))
    return result


def plot(title, xlabel, ylabel, first_metric, second_metric, label_first_graph, label_seconds_graph, savepath):
    plt.rcParams.update({'font.size': 22})
    plt.rcParams.update({'font.weight': 'bold'})

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(axis='y', linestyle='-')
    plt.plot(first_metric, color='blue', label=label_first_graph, alpha=0.7)
    plt.plot(second_metric, color='red', label=label_seconds_graph, alpha=0.7)
    plt.legend()

    plt.savefig(savepath)
    plt.clf()
    plt.close()


def plot_training_metrics(metrics_path, savepath, num_epochs=None):
    metrics = torch.load(metrics_path)
    train_losses = metrics['train_losses']
    test_losses = metrics['test_losses']
    train_accuracies_audio = metrics['train_accuracies_audio']
    train_accuracies_video = metrics['train_accuracies_video']
    test_accuracies_audio = metrics['test_accuracies_audio']
    test_accuracies_video = metrics['test_accuracies_video']

    if num_epochs is not None:
        num_batches = len(train_losses) // num_epochs
        train_losses = get_epoch_means(train_losses, num_batches)

    # Plot losses
    title = 'Train and evaluation losses per epoch'
    ylabel = 'Loss'
    xlabel = 'Epoch'
    label_first_graph = 'Train Loss'
    label_second_graph = 'Validation Loss'
    path = os.path.join(savepath, 'train_losses.png')
    plot(title, xlabel, ylabel, train_losses, test_losses, label_first_graph, label_second_graph, path)

    # Plot audio accuracies
    title = 'Audio accuracy per epoch'
    ylabel = 'Audio accuracy'
    xlabel = 'Epoch'
    label_first_graph = 'Train Accuracy'
    label_second_graph = 'Validation Accuracy'
    path = os.path.join(savepath, 'accuracies_audio.png')
    plot(title, xlabel, ylabel, train_accuracies_audio, test_accuracies_audio, label_first_graph, label_second_graph, path)

    # Plot video accuracies
    title = 'Video accuracy per epoch'
    ylabel = 'Video accuracy'
    path = os.path.join(savepath, 'accuracies_video.png')
    plot(title, xlabel, ylabel, train_accuracies_video, test_accuracies_video, label_first_graph, label_second_graph, path)

# Create Audio Video CLIP evaluation video =============================================================================


def create_sample_video(video_path, savepath, audio_starting_point, audio_duration, video_duration, sample_rate):
    video = VideoFileClip(video_path)
    video = video.subclip(audio_starting_point, audio_starting_point + audio_duration)
    audio = video.audio

    audio_path = os.path.join(savepath, 'audio.wav')
    if not os.path.isfile(audio_path):
        print(f'Saving audio to {audio_path}')
        audio.write_audiofile(audio_path, fps=sample_rate, ffmpeg_params=["-ac", "1"])
    video_path = os.path.join(savepath, 'full_video.mp4')
    if not os.path.isfile(video_path):
        print(f'Saving video to {video_path}')
        video.write_videofile(video_path)

    # start = randrange(audio_duration - video_duration)
    start = 500
    end = start + video_duration
    video = video.subclip(start, end)

    video_path = os.path.join(savepath, 'original_video.mp4')
    if not os.path.isfile(video_path):
        print(f'Saving original video to {video_path}')
        video.write_videofile(video_path)

    audio_path = os.path.join(savepath, 'original_audio.wav')
    if not os.path.isfile(audio_path):
        print(f'Saving original audio to {audio_path}')
        video.audio.write_audiofile(audio_path, fps=sample_rate)


def make_video_embeddings(video_path, model, fps, num_frames, batch_size, num_workers, prefetch_factor):
    video_dataset = VideoDataset(video_path, fps, num_frames)
    video_dataloader = DataLoader(video_dataset, batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor)

    print('Encoding video samples...')
    result = []
    with torch.no_grad():
        for video_batch in tqdm(video_dataloader):
            video_batch = video_batch.to(device)
            output = model.forward_video(video_batch)
            result.append(output)
    result = torch.cat(result)

    return result


def make_audio_embeddings(audio_path, model, batch_size, num_workers, prefetch_factor):
    audio_dataset = AudioDataset(audio_path)
    audio_dataloader = DataLoader(audio_dataset, batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor)

    print('Encoding audio samples...')
    result = []
    with torch.no_grad():
        for audio_batch in tqdm(audio_dataloader):
            audio_batch = audio_batch.to(device)
            output = model.forward_audio(audio_batch)
            result.append(output)
    result = torch.cat(result)

    return result


def cosine_similarity(embedding1, embedding2, use_softmax=True):
    # normalizing
    embedding1 = embedding1 / embedding1.norm(dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(dim=-1, keepdim=True)

    # cosine similarity
    similarity = (embedding1 @ embedding2.t()).cpu()
    if use_softmax:
        similarity = torch.nn.functional.softmax(similarity, dim=-1)

    return similarity


def create_new_audio(video_path, savepath, audio_path, indices):
    audio_video = VideoFileClip(audio_path)
    video = VideoFileClip(video_path)
    result = []
    for i, idx in enumerate(indices):
        curr_video = video.subclip(i, i + 1)
        audio_clip = audio_video.subclip(idx, idx + 1)
        curr_video.audio = audio_clip.audio
        result.append(curr_video)
    new_video = concatenate_videoclips(result)
    video_path = os.path.join(savepath, 'clip_video.mp4')
    new_video.write_videofile(video_path)


def compare_audio_video(similarities):
    diag = torch.diagonal(similarities)
    print(diag.shape)
    mean = diag.mean()
    return mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='videos/resampled/minecraft_video_1_final.mp4')
    parser.add_argument('--savepath', type=str, default='results/minecraft_avclip/additional_tests/clip_10layers_bn_no_activation_lr_1e-2')
    parser.add_argument('--model_path', type=str, default='checkpoints/evaluation_checkpoints/tests/clip_10layers_bn_no_activation_lr_1e-2.pth')
    parser.add_argument('--model_cfg', type=str, default='minecraft_audio_video_clip/configs/10_layers_no_activation.yaml')
    parser.add_argument('--train_param_path', type=str, default='checkpoints/evaluation_checkpoints/tests/clip_10layers_bn_no_activation_lr_1e-2_train_params.pth')
    parser.add_argument('--audio_starting_point', type=int, default=1200)
    parser.add_argument('--audio_duration', type=int, default=1200)
    parser.add_argument('--video_duration', type=int, default=60)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--fps', type=int, default=32)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--only_plotting', type=bool, default=False)
    args = parser.parse_args()

    image_savepath = os.path.join(args.savepath, 'train_metrics')
    os.makedirs(image_savepath, exist_ok=True)
    if not args.only_plotting:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        create_sample_video(args.video_path, args.savepath, args.audio_starting_point, args.audio_duration, args.video_duration, args.sample_rate)

        model = load_model(args.model_path, args.model_cfg, device)
        video_path = os.path.join(args.savepath, 'original_video.mp4')
        audio_path = os.path.join(args.savepath, 'full_video.mp4')

        video_embeddings = make_video_embeddings(video_path, model, args.fps, args.num_frames, args.batch_size, args.num_workers, args.prefetch_factor)
        audio_embeddings = make_audio_embeddings(audio_path, model, args.batch_size, args.num_workers, args.prefetch_factor)
        print(f'Encoded video samples with shape {video_embeddings.shape}')
        print(f'Encoded audio samples with shape {audio_embeddings.shape}')

        print('Calculating indices of matching audio sample for each video sample')
        similarities = cosine_similarity(video_embeddings, audio_embeddings)
        print(similarities.shape)
        indices = torch.argmax(similarities, dim=-1).numpy()
        print(indices.shape)
        print(indices)

        print('Creating new audio track based on matching audio samples')

        create_new_audio(video_path, args.savepath, audio_path, indices)

        print('Calculating average cosine similarity between original video and audio')
        audio_path = os.path.join(args.savepath, 'original_audio.wav')
        audio_embeddings = make_audio_embeddings(audio_path, model, args.batch_size, args.num_workers, args.prefetch_factor)
        similarities = cosine_similarity(video_embeddings, audio_embeddings, False)
        mean = compare_audio_video(similarities)
        print(f'Mean cosine similarity of the original audio and video is {mean}')

    print('Creating plots for the training metrics')
    plot_training_metrics(args.train_param_path, image_savepath, args.num_epochs)
