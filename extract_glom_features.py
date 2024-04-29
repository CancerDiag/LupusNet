import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5_lupus import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline, resnet18_baseline, vits_small, vits_for_large
from models.resnet_custom import vit_s_new, resnet50_new, ctranspath
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide

from StainNet.models_st import StainNet, ResnetGenerator

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True,
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'

	# stain_Net = StainNet().cuda()
	# stain_Net.load_state_dict(torch.load("/home/ekansh.chauhan/CLAM/StainNet/checkpoints/aligned_histopathology_dataset/StainNet-Public_layer3_ch32.pth"))
	# stain_Net = nn.DataParallel(stain_Net)
	# stain_Net.eval()

	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			# batch = stain_Net(batch)
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'

	return output_path


def compute_w_loader_for_4096(file_path, output_path, bag_name, dir_names, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'

	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			features = model(batch)

			# features = features.cpu().numpy()

			# asset_dict = {
			# 	'features_cls256': features_cls256.numpy(),
			# 	'features_mean256': features_mean256.numpy(),
			# 	'features_cls4k': features_cls4k.numpy(),
			# 	'features_mean256_cls4k': features_mean256_cls4k.numpy()
			# }
			output_paths = []
			for out_idx, out_dir in enumerate(output_path):
				out_dir = os.path.join(out_dir, bag_name)
				asset_dict = {'features': features[dir_names[out_idx]].detach().cpu().numpy(), 'coords': coords}
				save_hdf5(out_dir, asset_dict, attr_dict= None, mode=mode)
				output_paths.append(out_dir)

			mode = 'a'

	return output_paths

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model_type', type=str, default=None)

args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)

	os.makedirs(args.feat_dir, exist_ok=True)
	out_dir_name = []
	dest_files = []

	if args.model_type == 'vit_4096':
		dir_names = ['features_cls256', 'features_mean256', 'features_cls4k', 'features_mean256_cls4k']
		for name in dir_names:
			path = os.path.join(args.feat_dir, name)
			os.makedirs(os.path.join(path, 'h5_files'), exist_ok=True)
			os.makedirs(os.path.join(path, 'pt_files'), exist_ok=True)
			dest_files.append(os.listdir(os.path.join(path, 'pt_files')))
			out_dir_name.append(os.path.join(path, 'h5_files'))

	if args.model_type != 'vit_4096':
		os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
		os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
		dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
		out_dir_name.append(os.path.join(args.feat_dir, 'h5_files'))

	print('loading model checkpoint')

	args.pretrained_norm = False

	if args.model_type == 'resnet_18':
		model = resnet18_baseline(pretrained=True)
	elif args.model_type == 'resnet_50':
		args.pretrained_norm = True
		model = resnet50_baseline(pretrained=True)
	elif args.model_type == 'resnet_50_BT':
		model = resnet50_new(pretrained=True)
	elif args.model_type == 'resnet_50_MoCoV2':
		model = resnet50_new(pretrained=True, progress=False, key="MoCoV2")
	elif args.model_type == 'resnet_50_SwAV':
		model = resnet50_new(pretrained=True, progress=False, key="SwAV")
	elif args.model_type == 'vit_256':
		model = vits_small(pretrained=True)
	elif args.model_type == 'vit_s_new':
		model = vit_s_new(pretrained=True)
	elif args.model_type == 'vit_4096':
		model = vits_for_large(pretrained=True)
	elif args.model_type == 'ctranspath':
		model = ctranspath(pretrained=True)

	model = model.to(device)
	print('model used: ', args.model_type)
	print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	model.eval()
	total = len(bags_dataset)
	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx][0].split(args.slide_ext)[0]
		slide_status = bags_dataset[bag_candidate_idx][1]

		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if args.model_type != 'vit_4096':
			if not args.no_auto_skip and slide_id+'.pt' in dest_files:
				print('skipped no_auto_skip and in dest files : {}'.format(slide_id))
				continue
			elif 'failed' in slide_status:
				print('skipped {}: {}'.format(slide_status, slide_id))
				continue
		else:
			if not args.no_auto_skip and slide_id+'.pt' in dest_files[0] and slide_id+'.pt' in dest_files[1] and slide_id+'.pt' in dest_files[2] and slide_id+'.pt' in dest_files[3]:
				print('skipped {}'.format(slide_id))
				continue
			

		time_start = time.time()

		wsi = openslide.open_slide(slide_file_path)			
			
		if args.model_type != 'vit_4096':
			output_path = os.path.join(out_dir_name[0], bag_name)
			try:
				output_path1 = compute_w_loader(h5_file_path, output_path, wsi, 
						model = model, batch_size = args.batch_size, verbose = 1, print_every = 10, 
						custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, pretrained= args.pretrained_norm)
				output_file_list = [output_path1]
			except:
				print('skipped due to error {}'.format(slide_id))
				continue

		else:
			output_file_list = compute_w_loader_for_4096(h5_file_path, out_dir_name, bag_name, dir_names, wsi, 
				       model = model, batch_size = args.batch_size, verbose = 1, print_every = 100, 
					   custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
			

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(bag_name, time_elapsed))

		for idx, output_file_path in enumerate(output_file_list):
			file = h5py.File(output_file_path, "r")

			features = file['features'][:]
			if idx == 0:
				print('coordinates size: ', file['coords'].shape)
			
			if args.model_type == 'vit_4096':
				print('features size: '+ str(dir_names[idx]) + ' :', features.shape)
			else:
				print('features size: ', features.shape)
				
			features = torch.from_numpy(features)
			bag_base, _ = os.path.splitext(bag_name)
			
			if args.model_type == 'vit_4096':
				torch.save(features, os.path.join(args.feat_dir, dir_names[idx], 'pt_files', bag_base+'.pt'))
			else:
				torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))