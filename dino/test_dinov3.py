import torch

REPO_DIR = '/home/frank/Documents/DLFinalProject/dinov3'

dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights='/home/frank/Documents/DLFinalProject/ogbench/dino/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
