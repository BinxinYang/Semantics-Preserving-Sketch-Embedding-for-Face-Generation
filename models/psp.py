"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
import numpy as np


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder_sketch_coarse, self.encoder_sketch_fine, self.encoder_face_fine = self.set_encoder()
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

		self.noise_vec = np.random.randn(1, 512).astype('float32')

	def set_encoder(self):
		encoder_sketch_coarse = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts, space='W', condition='sketch')
		encoder_sketch_fine = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts, space='W_plus', condition='sketch')
		encoder_face_fine = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts, space='W_plus', condition='face')

		return encoder_sketch_coarse,encoder_sketch_fine,encoder_face_fine

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder_sketch_coarse.load_state_dict(get_keys(ckpt, 'encoder_sketch_coarse'), strict=True)
			self.encoder_sketch_fine.load_state_dict(get_keys(ckpt, 'encoder_sketch_fine'), strict=True)
			self.encoder_face_fine.load_state_dict(get_keys(ckpt, 'encoder_face_fine'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			if self.opts.coarse_encoder_checkpoint_path is not None:
				print('Loading coarse encoder')
				ckpt = torch.load(self.opts.coarse_encoder_checkpoint_path, map_location='cpu')
				self.encoder_sketch_coarse.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

	def forward(self, x, y, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder_sketch_coarse(x)
			codes = self.decoder.style(codes)
			codes = codes.unsqueeze(1).repeat(1,18,1)

			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		codes_sketch_fine=self.encoder_sketch_fine(x)
		if self.opts.random_appearance:
			if self.opts.fixed_random_appearance:
				noise_vec=self.noise_vec
			else:
				noise_vec = np.random.randn(x.shape[0], 512).astype('float32')
			noise_vec=torch.from_numpy(noise_vec).to("cuda")
			real_vec=self.decoder.style(noise_vec)
			real_vec=real_vec.view(x.shape[0],1,512)
			codes_face_fine=real_vec.repeat(1, 18, 1)
			codes=torch.cat((codes_sketch_fine[:,:8,:]+codes[:,:8,:],codes_face_fine[:,8:,:]),dim=1)
		else:
			codes_face_fine=self.encoder_face_fine(self.face_pool(y))
			codes=codes+torch.cat((codes_sketch_fine[:,:8,:],codes_face_fine[:,8:,:]),dim=1)
		
		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)
		images_1024=images
		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, images_1024, result_latent
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
