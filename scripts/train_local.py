# Imports
import torch
from minecraft_audio_video_clip.source import train_model, EmbeddingLoader
import matplotlib.pyplot as plt
import time

# Paths
train_data_path = 'datasets/embeddings/extended/train_extended.npz'
test_data_path = 'datasets/embeddings/extended/test_extended.npz'
model_path = 'checkpoints/model/10_layers.pth'
cfg_path = 'minecraft_audio_video_clip/configs/10_layers.yaml'
# train_param_path = 'checkpoints/model/test_1_train_params.pth'
train_param_path = None
savepath = 'checkpoints/evaluation_checkpoints/tests/clip_10layers_bn_no_activation_bs_2048.pth'
param_path = 'checkpoints/evaluation_checkpoints/tests/clip_10layers_bn_no_activation_bs_2048_train_params.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')
dataloader = EmbeddingLoader(1024, train_data_path, test_data_path)

print('Loaded {} training samples in {} batches'.format(dataloader.len_train_set(), dataloader.len_train_batches()))
start = time.time()
model = train_model(model_path, dataloader, savepath, cfg_path=cfg_path, param_path=train_param_path, lr=1e-3, num_epochs=300)
# model = train_model(model_path, dataloader, savepath, cfg_path=cfg_path, param_path=train_param_path, lr=0.001, start_epochs=0, num_epochs=100)
end = time.time()

print('Time taken to train model: {}'.format(end - start))

chkpt = torch.load(param_path, map_location=device)
losses = chkpt['train_losses']
plt.plot(losses)
plt.show()
