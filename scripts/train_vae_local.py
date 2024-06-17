from minecraft_audio_video_clip.source.dataloading.datasets import EmbeddingsDataset
from minecraft_audio_video_clip.source.cvae.cvae import load_cvae
from minecraft_audio_video_clip.source.cvae.train_cvae import train_cvae
import torch
from torch.utils.data import DataLoader

# Paths
model_cfg = 'minecraft_audio_video_clip/configs/cvae/cvae.yaml'
train_data_path = 'datasets/audio_embeddings/train_audio_embeddings_projection.npz'
val_data_path = 'datasets/audio_embeddings/test_audio_embeddings_projection.npz'
savepath = 'checkpoints/cvae/audio_prior_projection.pth'

# Params
batch_size = 2048
prefetch_factor = 2
num_workers = 2
epochs = 500
lr = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_cvae(model_cfg, device=device)

train_dataset = EmbeddingsDataset(train_data_path)
val_dataset = EmbeddingsDataset(val_data_path)

print(len(train_dataset))
print(len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=prefetch_factor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor)

train_cvae(model, train_loader, val_loader, savepath, lr, epochs)
