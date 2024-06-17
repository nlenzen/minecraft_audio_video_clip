# Imports
import os
import sys
import hashlib
import math

sys.path.append(os.getcwd())

import numpy as np
import torch
from torchaudio import load
from torch.utils.data import Dataset
from minecraft_audio_video_clip.source.load import load_model
from mineclip import MineCLIP
from minecraft_audio_video_clip.source.preprocess import make_features


def get_file_num_index(idx, sizes):
    num_file = 0
    n = 0
    while idx >= n:
        n += sizes[num_file]
        num_file += 1
    num_file -= 1
    sample_idx = idx - (n - sizes[num_file])

    return num_file, sample_idx


class dataset(Dataset):
    def __init__(self, wavefiles, sr):
        self.wavefiles = wavefiles
        self.sr = sr
        self.file_sizes = []
        for wavefile in self.wavefiles:
            data, _ = load(wavefile)
            l = int(data.shape[1] / self.sr)
            l = (l - 1) * 4 + 1
            self.file_sizes.append(l)

    def __len__(self):
        len = 0
        for size in self.file_sizes:
            len += size
        return len

    def __getitem__(self, idx):
        num_file, sample_idx = get_file_num_index(idx, self.file_sizes)
        data, _ = load(self.wavefiles[num_file]).squeeze(0)
        start = int(sample_idx * 0.25 * self.sr)
        end = int(start + self.sr)

        item = data[start:end]
        features = make_features(item, sr=self.sr)

        return features


MINECLIP_CONFIG = {
    'arch': "vit_base_p16_fz.v2.t2",
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': "attn.d2.nh8.glusw",
    'resolution': [160, 256],
    'ckpt': {
        'path': "STEVE-1/data/weights/mineclip/attn.pth",
        'checksum': 'b5ece9198337cfd117a3bfbd921e56da'
    }
}


def load_mineclip(cfg, device):
    cfg = cfg.copy()
    ckpt = cfg.pop("ckpt")
    assert hashlib.md5(open(ckpt['path'], "rb").read()).hexdigest() == ckpt['checksum'], "broken ckpt"

    model = MineCLIP(**cfg).to(device)
    model.load_ckpt(ckpt['path'], strict=True)
    return model


# Paths
audio_embeddings = 'datasets/from_videos/encoded_subfiles/train_audio_embeddings.npz'
model_path = 'checkpoints/model_mineclip/no_act_L1_extended_dataset.pth'
model_cfg = 'minecraft_audio_video_clip/configs/mineclip/10_layers_mineclip_no_activation.yaml'

# Params
batch_size = 2048


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mineclip = load_mineclip(MINECLIP_CONFIG, device)
model = load_model(model_path, model_cfg, device)

texts = ['dig as far as possible', 'get dirt', 'go explore', 'go swimming', 'go underwater', 'chop a tree', 'break leaves']
wavefiles = ['videos/resampled/minecraft_video_1_final.wav',
             'videos/resampled/video2/output0.wav',
             'videos/resampled/video2/output1.wav',
             'videos/resampled/video2/output3.wav',
             'videos/resampled/video2/output4.wav',
             'videos/resampled/video2/output5.wav',
             'videos/resampled/video3/output0.wav',
             'videos/resampled/video3/output1.wav',
             'videos/resampled/video3/output2.wav',
             'videos/resampled/video3/output3.wav',
             'videos/resampled/video3/output4.wav',
             'videos/resampled/video3/output5.wav']

with torch.cuda.amp.autocast():
    text_embeds = mineclip.encode_text(texts).detach().to(torch.float32)

embeddings = np.load(audio_embeddings)['audio']
batch_num = math.ceil(len(embeddings) / batch_size)

# data = dataset(wavefiles, sr=16000)
# dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=2)

audio_embeds = []
for i in range(batch_num):
    start = i * batch_size
    end = start + batch_size
    batch = torch.tensor(embeddings[start:end]).to(device)
    with torch.no_grad():
        emb = model.project_audio_embeddings(batch)
    audio_embeds.append(emb)

audio_embeds = torch.cat(audio_embeds, dim=0).to(torch.float32)

# Compute cosine similarity
text_embed = text_embeds / text_embeds.norm(dim=1, keepdim=True)
audio_embeds = audio_embeds / audio_embeds.norm(dim=1, keepdim=True)

similarities = (text_embeds @ audio_embeds.T).softmax(dim=-1).cpu()

print('Saving top files')
torch.set_printoptions(precision=10)
savepath = "datasets/mineclip_extended_no_act"
for i, line in enumerate(similarities):
    top_k = line.topk(5)
    print(texts[i] + ': {}, based on {}'.format(top_k.indices, top_k.values))

