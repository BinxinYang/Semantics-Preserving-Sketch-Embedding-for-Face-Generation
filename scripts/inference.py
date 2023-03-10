import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
import random


def run():
	test_opts = TestOptions().parse()

	if test_opts.resize_factors is not None:
		assert len(test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
		                                'downsampling_{}'.format(test_opts.resize_factors))
		out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
										'downsampling_{}'.format(test_opts.resize_factors))
	else:
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
		out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
		out_path_latents = os.path.join(test_opts.exp_dir, 'inference_latents')
		out_path_reference = os.path.join(test_opts.exp_dir, 'inference_reference')
	os.makedirs(out_path_results, exist_ok=True)
	os.makedirs(out_path_coupled, exist_ok=True)
	os.makedirs(out_path_latents, exist_ok=True)
	os.makedirs(out_path_reference, exist_ok=True)
	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()
	print('Loading dataset for {}'.format(opts.dataset_type))
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	dataset = InferenceDataset(source_root=opts.data_path,
							   target_root=opts.target_path,
							   source_transform=transforms_dict['transform_source'],
							   target_transform=transforms_dict['transform_gt_train'],
	                           opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=True)

	if opts.n_images is None:
		opts.n_images = len(dataset)
	
	global_i = 0
	global_time = []
	for input_batch in tqdm(dataloader):
		if global_i >= opts.n_images:
			break
		with torch.no_grad():
			input_cuda,target_cuda=input_batch
			input_cuda = input_cuda.cuda().float()
			target_cuda = target_cuda.cuda().float()
			tic = time.time()
			result_batch,latent_batch = run_on_batch(input_cuda,target_cuda, net, opts)
			toc = time.time()
			global_time.append(toc - tic)
		for i in range(opts.test_batch_size):
			result = tensor2im(result_batch[i])
			reference=tensor2im(target_cuda[i])
			latent = latent_batch[i]
			im_path = dataset.source_paths[global_i]
			if opts.couple_outputs or global_i % 100 == 0:
				# print(input_batch[i].shape)
				input_im = log_input_image(input_cuda[i], opts)
				resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
				if opts.resize_factors is not None:
					# for super resolution, save the original, down-sampled, and output
					source = Image.open(im_path)
					res = np.concatenate([np.array(source.resize(resize_amount)),
										  np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
										  np.array(result.resize(resize_amount))], axis=1)
				else:
					# otherwise, save the original and output
					res = np.concatenate([np.array(input_im.resize(resize_amount)),
										  np.array(result.resize(resize_amount)),
                                          np.array(reference.resize(resize_amount))], axis=1)
				Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

			im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
			reference_save_path = os.path.join(out_path_reference, os.path.basename(im_path))
			latent_save_path=os.path.join(out_path_latents, os.path.basename(im_path)[:-4]+'.pt')
			Image.fromarray(np.array(result)).save(im_save_path)
			Image.fromarray(np.array(reference)).save(reference_save_path)
			torch.save(latent,latent_save_path)
			global_i += 1

	stats_path = os.path.join(opts.exp_dir, 'stats.txt')
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)

	with open(stats_path, 'w') as f:
		f.write(result_str)


def run_on_batch(inputs,targets, net, opts):
	if opts.latent_mask is None:
		result_batch,result_batch_noise,latent_batch = net(inputs[:,1:,:,:],targets,randomize_noise=False,return_latents=True, resize=opts.resize_outputs)
	else:
		latent_mask = [int(l) for l in opts.latent_mask.split(",")]
		result_batch = []
		latent_batch = []
		for image_idx, input_image in enumerate(inputs[:,1:,:,:]):
			# get latent vector to inject into our input image
			vec_to_inject = np.random.randn(1, 512).astype('float32')
			_, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
									input_code=True,
									return_latents=True)
			res,_,latent_to_save = net(input_image.unsqueeze(0).to("cuda").float(),targets,
			          latent_mask=latent_mask,
			          inject_latent=latent_to_inject,
			          return_latents=True,
					  resize=opts.resize_outputs)
			result_batch.append(res)
			latent_batch.append(latent_to_save)
		result_batch = torch.cat(result_batch, dim=0)
		latent_batch = torch.cat(latent_batch, dim=0)
	return result_batch,latent_batch


if __name__ == '__main__':
	run()
