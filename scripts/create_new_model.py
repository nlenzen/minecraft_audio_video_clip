import torch
from minecraft_audio_video_clip.source import init_model, save_model

conf_path = 'minecraft_audio_video_clip/configs/projection_bn.yaml'
video_enc_path = 'checkpoints/mineclip/attn.pth'
audio_enc_path = 'checkpoints/ast/audioset_10_10_0.4593.pth'
savepath = 'checkpoints/model/base_projection_bn.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = init_model(video_enc_path, audio_enc_path, conf_path, device)
save_model(model, savepath)
