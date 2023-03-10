import torch
import os
from torch import nn
w_average='/home/eeid/tennyson/code/pixel2style2pixel-master-TMM/latent_avg.pt'
w_hat='/home/eeid/tennyson/code/pixel2style2pixel-master-TMM/mean_latent_stylegan_30000_hat_new.pt'
w_average=torch.load(w_average).unsqueeze(0)
w_hat=torch.load(w_hat).cuda()
latent=w_hat-w_average
print(latent.shape)
print(w_average.shape,w_hat.shape)
latent=latent.unsqueeze(0)
loss=torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]
print(loss)
