import numpy as np
import os
import sys

sys.path.append(os.getcwd())


def fix_datasets(datasets, savepath, audio_embed_dim, video_embed_dim):
    for dataset in datasets:
        path = os.path.join(savepath, dataset)
        data = np.load(path)
        audio = data['audio']
        audio = np.reshape(audio, (-1, audio_embed_dim))
        video = data['video']
        video = np.reshape(video, (-1, video_embed_dim))

        np.savez(path, audio=audio, video=video)


def combine_datasets(datasets, savepath, name):
    num_samples = 0
    video_data = []
    audio_data = []
    for dataset in datasets:
        print('Processing {}'.format(dataset))
        path = os.path.join(savepath, dataset)
        data = np.load(path)
        audio = data['audio']
        video = data['video']
        print(f'Length of video: {video.shape}')
        print(f'Length of audio: {audio.shape}')
        if len(audio) == len(video):
            num_samples += video.shape[0]
            video_data.append(video)
            audio_data.append(audio)
        else:
            print('Error: length of video and audio do not match in dataset {}'.format(dataset))
            print(video)

    print('Samples: {}'.format(num_samples))
    video_data = np.concatenate(video_data, axis=0)
    audio_data = np.concatenate(audio_data, axis=0)
    print('Amount of combined samples: {}'.format(video_data.shape))
    path = os.path.join(savepath, name)
    np.savez(path, audio=audio_data, video=video_data)


train_datastes = [
    'video_1_train_embeddings.npz',
    'train_embeddings_video_2.npz',
    'train_embeddings_video_3.npz',
    'train_embeddings_video_4.npz',
    'train_embeddings_video_5.npz',
    'train_embeddings_video_6.npz',
    'train_embeddings_video_7.npz',
    'train_embeddings_video_8.npz',
    'train_embeddings_video_9.npz',
    'train_embeddings_video_10.npz',
    'train_embeddings_video_11.npz',
    'train_embeddings_video_12.npz',
    'train_embeddings_video_13.npz',
    'train_dirt_1.npz',
    'train_dirt_2.npz',
    'train_wood.npz',
    'train_sand.npz',
    'train_snow.npz',
    'train_water.npz',
    'train_seeds.npz',
    'train_leaves_1.npz',
    'train_leaves_2.npz',
    'train_stone.npz',
    'train_dirt_3.npz',
    'train_cobblestone_1.npz',
    'train_cobblestone_2.npz',
    'train_place_dirt.npz',
    'train_place_sand.npz',
    'train_wood.npz',
]

test_datastes = [
    'video_1_test_embeddings.npz',
    'test_embeddings_video_2.npz',
    'test_embeddings_video_3.npz',
    'test_embeddings_video_4.npz',
    'test_embeddings_video_5.npz',
    'test_embeddings_video_6.npz',
    'test_embeddings_video_7.npz',
    'test_embeddings_video_8.npz',
    'test_embeddings_video_9.npz',
    'test_embeddings_video_10.npz',
    'test_embeddings_video_11.npz',
    'test_embeddings_video_12.npz',
    'test_embeddings_video_13.npz',
    'test_dirt_1.npz',
    'test_water.npz',
    'test_seeds.npz',
    'test_leaves_1.npz',
    'test_leaves_2.npz',
    'test_dirt_2.npz',
    'test_wood.npz',
    'test_sand.npz',
    'test_snow.npz',
    'test_stone.npz',
    'test_dirt_3.npz',
    'test_cobblestone_1.npz',
    'test_cobblestone_2.npz',
    'test_place_dirt.npz',
    'test_place_sand.npz',
    'test_wood.npz',
]

savepath = 'datasets/embeddings/extended'

combine_datasets(train_datastes, savepath, 'train_extended.npz')
combine_datasets(test_datastes, savepath, 'test_extended.npz')

# fix_datasets(train_datastes, savepath, 527, 512)
# fix_datasets(test_datastes, savepath, 527, 512)
