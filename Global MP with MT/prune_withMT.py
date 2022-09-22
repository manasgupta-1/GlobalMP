import argparse
import os
import random
import shutil
import time
import warnings
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from numpy import genfromtxt
import numpy as np
import pandas as pd
# import wandb
import datetime
import torch.nn.functional as F
from model_list import LeNet5Caffe, WideResNet, ResNet50, MobileNetV2
from LabelSmoothing import LabelSmoothing

wrn_models = {}
wrn_models['wrn-22-8'] = (WideResNet, [22, 8, 10, 0.3])
wrn_models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.3])
wrn_models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.3])
wrn_models['wrn-28-8'] = (WideResNet, [28, 8, 10, 0.3])

model_names = ['resnet50_imagenet', 'mobilenet_v2', 'mobilenet_v2_cifar10', 'efficientnet-b2', 'lenet5caffe', 'wrn-22-8', 'wrn-16-8', 'wrn-28-8', 'wrn-16-10']
data_sets = ['imagenet', 'mnist', 'cifar10']
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-set', default='imagenet', choices=data_sets, help='Datasets: ' + ' | '.join(data_sets))
parser.add_argument('-a', '--arch', default='resnet50_imagenet', choices=model_names, help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('-j', '--workers', default=20, help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to finetune')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epochs', default=0, type=int, help='numbers of epochs to warmup')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='total batch size for all GPUs')
parser.add_argument('--lr', '--learning-rate', default=0.1024, type=float)
parser.add_argument('--momentum', default=0.875, type=float, help='momentum')
parser.add_argument('--weight-decay', type=float, default=0.00000700000000)
parser.add_argument('--smoothing', default=0.1, help='Smoothing factor')
parser.add_argument('-p', '--print-freq', default=500, type=int, help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, help='path to model checkpoint (default: '')')
parser.add_argument('--log-dir', default='./', type=str, help='path to latest checkpoint (default: ./)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8088', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu-prune', default=torch.device('cuda:0'), type=int, help='GPU id to use for prune base.')
parser.add_argument('--gpu', default=None, help='GPU id to use for finetuning.')
parser.add_argument('-mp','--multiprocessing-distributed', action='store_false',
					help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the '
						'fastest way to use PyTorch for either single node or multi node data parallel training')
parser.add_argument('--model', default='./prunedmodel.pth.tar', type=str, help='path to original pruned model')
parser.add_argument('--mask', default='./weight_mask.npy', type=str, help='path to weight-mask file')
parser.add_argument('-n', '--nesterov', action='store_true', help='Use nesterov by default unless switched off')
parser.add_argument('--sparsity', default='0.90', type=float, help='sparsity target')
parser.add_argument('-mt', '--min-threshold', default='0.05', type=float, help='Minimum number of weights that every layer should have')
parser.add_argument('--input-data', default='./prune_log.csv', help='Input data path')
best_acc1 = 0
min_weight_target = 0
val_loader = 0
def step():
	print("Starting prune base")
	args = parser.parse_args()
	layer_index = 0
	if args.sparsity == 0.0:
		print("Enter sparsity target\nExiting")
		exit()
	## CREATE MODEL
	print("=> creating model '{}'".format(args.arch))
	if args.arch.startswith("efficientnet"):
		model = EfficientNet.from_name(args.arch)
	elif args.arch == "mobilenet_v2":
		model = models.__dict__[args.arch]()
	elif args.arch.startswith("wrn"):
		cls, cls_args = wrn_models[args.arch]
		model = cls(*(cls_args))
		args.nesterov = True
	elif args.arch == 'lenet5caffe':
		model = LeNet5Caffe()
	elif args.arch == 'resnet50_cifar10':
		model = ResNet50()
	elif args.arch == "resnet50_imagenet":
		model = models.__dict__['resnet50']()
	elif args.arch == "mobilenet_v2_cifar10":
		model = MobileNetV2()
	## Move to GPU
	torch.cuda.empty_cache()
	model = model.to(args.gpu_prune)
	print("Using GPUs: {}".format(args.gpu_prune))
	criterion = nn.CrossEntropyLoss().cuda(args.gpu_prune)
	## LOAD DATA
	if args.data_set == 'imagenet':
		data_path = '/datasets/imagenet/val/'
		val_dataset = datasets.ImageFolder(data_path, transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
	elif args.data_set == 'mnist':
		data_path = '/data/'
		val_dataset = datasets.MNIST(data_path, train=False, transform=transforms.Compose([
							transforms.ToTensor(), 
							transforms.Normalize((0.1307,), (0.3081,))]))			
	elif args.data_set == 'cifar10':
		data_path = '/data/'
		val_dataset = datasets.CIFAR10(data_path, train=False, transform=transforms.Compose([
							transforms.ToTensor(), 
							transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))		
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
	cudnn.benchmark = True
	## LOAD STARTING CHECKPOINT
	if args.data_set == 'imagenet' and args.arch == 'resnet50_imagenet':
		args.resume = 'ResNet50_Imagenet_77.04_unwrapped.pth'
		pre_acc = 77.04
	elif args.data_set == 'cifar10' and args.arch == 'wrn-28-8':
		args.resume = 'WRN-28-8-Dense-New-New.pth.tar'
		pre_acc = 96.04	
	elif args.data_set == 'mnist' and args.arch == 'lenet5caffe':
		args.resume = 'mnist_lenet5_99.08_r2.pt'
		pre_acc = 99.08
	elif args.data_set == 'cifar10' and args.arch.startswith("wrn"):
		args.resume = 'WRN-22-8_94.0.pt'
		pre_acc = 94.0
	elif args.data_set == 'cifar10' and args.arch == 'resnet50_cifar10':
		args.resume = 'ResNet50_CIFAR10_95.16_unwrapped.pth'
		pre_acc = 95.16	
	elif args.data_set == 'cifar10' and args.arch == 'mobilenet_v2_cifar10':
		pre_acc = float(args.resume[-8:-3])	
	if os.path.isfile(args.resume):
		checkpoint = torch.load(args.resume, map_location='cuda:0')
		model.load_state_dict(checkpoint)
		print("=> loaded checkpoint '{}'".format(args.resume))
	else:
		print("=> no checkpoint found at '{}'".format(args.resume))

	print("Starting acc is ", float(validate_pruning(val_loader, model, criterion, args)))		
	columns = ['Layer Index','Layer Weights','Pre Accuracy','Weights Pruned','% Pruning','Post Accuracy']
	df = pd.DataFrame(columns=columns)
	layer_index=0
	mask_arr = []
	log_path = "{}/prune_log.csv".format(args.log_dir)
	#Calculate weight_threshold for global pruning
	weights = torch.empty(1, device=args.gpu_prune)
	print(weights)
	for child in list(model.modules()):
		if (isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)):
			layer_weights = torch.flatten(child.weight.data.abs())
			print(layer_weights.size())
			weights = torch.cat((weights, layer_weights))
	print(weights.size())
	num_weights = weights.size()[0]
	print(num_weights)
	min_weight_target = int(num_weights * args.min_threshold / 100)
	k = int (args.sparsity * num_weights)
	print(k)
	if k==0: k=1
	weight_threshold = torch.kthvalue(weights,k).values
	print("Mean of weights is ", torch.mean(weights))
	print("Weight threshold is ",weight_threshold)
	for child in list(model.modules()):
		if (isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)):
			# Do pruning
			num_pruned, num_weights, mask = prune(child, weight_threshold)
			mask_arr.append(mask)
			pctpruned = 100*num_pruned/num_weights
			print("Pruned layer: {} Pct pruned: {}".format(layer_index, pctpruned))
			layer_index += 1
			#Log results
			df2 = pd.DataFrame([[layer_index, num_weights, pre_acc, num_pruned, pctpruned, "NA"]], columns=columns)
			df = df.append(df2, ignore_index=True)
			df.to_csv(log_path)
	#Save log
	post_acc = float(validate_pruning(val_loader, model, criterion, args))
	df2 = pd.DataFrame([["", "", "", "", "", post_acc]], columns=columns)
	df = df.append(df2, ignore_index=True)
	df.to_csv(log_path)
	return min_weight_target, val_loader #to pass to step2() and step3()

def prune(m, weight_threshold):
	"""
	Prunes a given layer and creates a weightmask for this layer
	Args:
		m: layer to prune
		weight_threshold: magnitude threshold used to do pruning
	"""
	num_weights = torch.numel(m.weight.data)
	weight_mask = torch.ge(m.weight.data.abs(), weight_threshold).type('torch.cuda.FloatTensor')
	m.weight.data *= weight_mask
	num_pruned = num_weights - torch.nonzero(weight_mask).size(0)
	return num_pruned, num_weights, weight_mask

def step2():
	print("Starting refine")
	args = parser.parse_args()

	#Get input from user
	if args.sparsity == 0.0:
		print("Enter sparsity target\nExiting")
		exit()
	print("min_weight_target =", min_weight_target)
	input = genfromtxt(args.input_data, delimiter=',', skip_header=1)
	layer_weights = input[:-1,2]
	total_weights = np.sum(layer_weights)
	layer_sparsities = input[:-1,5] / 100
	print(layer_sparsities)
	#Set layer sparsity to 0 if weights < min_weight_target
	layer_sparsities[np.where(layer_weights < min_weight_target)] = 0
	print(layer_sparsities)
	total_sparsity = np.sum(layer_weights * layer_sparsities) / total_weights
	#Scale layer sparsities to meet required total sparsity target
	total_sparsity_target = float(args.sparsity)
	layer_sparsities = layer_sparsities / total_sparsity * total_sparsity_target
	total_sparsity = np.sum(layer_weights * layer_sparsities) / total_weights

	df = pd.DataFrame([layer_sparsities])
	print(total_sparsity)
	# Check if for any layer num_weights_left < min_weight_target. If so then set those sparsities to max limit such that 
	# num_weights_left = min_weight_target and iteratively increase other sparsities until total sparsity = total sparsity target
	if np.any((1-layer_sparsities)*layer_weights < min_weight_target):
		less_weight_indices = np.where(((1-layer_sparsities)*layer_weights < min_weight_target) & (layer_weights > min_weight_target))
		layer_sparsities[less_weight_indices] = 1 - min_weight_target / layer_weights[less_weight_indices]
		total_sparsity = np.sum(layer_weights * layer_sparsities) / total_weights
		df = df.append(pd.DataFrame([layer_sparsities]), ignore_index=True)
		print(total_sparsity)
		while(total_sparsity < total_sparsity_target):
			layer_sparsities = layer_sparsities / total_sparsity * total_sparsity_target
			less_weight_indices = np.where(((1-layer_sparsities)*layer_weights < min_weight_target) & (layer_weights > min_weight_target))
			layer_sparsities[less_weight_indices] = 1 - min_weight_target / layer_weights[less_weight_indices]
			total_sparsity = np.sum(layer_weights * layer_sparsities) / total_weights
			df = df.append(pd.DataFrame([layer_sparsities]), ignore_index=True)
			print(total_sparsity)

	#Save final values
	print("Final weight counts ",(1-layer_sparsities)*layer_weights)
	np.savetxt("layer_sparsities.csv", layer_sparsities, delimiter=",")
	# df.to_csv('layer_sparsities_log.csv')

def step3():
	print("Starting prune refined")
	args = parser.parse_args()
	layer_index = 0
	## CREATE MODEL
	print("=> creating model '{}'".format(args.arch))
	if args.arch.startswith("efficientnet"):
		model = EfficientNet.from_name(args.arch)
	elif args.arch == "mobilenet_v2":
		model = models.__dict__[args.arch]()
	elif args.arch.startswith("wrn"):
		cls, cls_args = wrn_models[args.arch]
		model = cls(*(cls_args))
	elif args.arch == 'lenet5caffe':
		model = LeNet5Caffe()
	elif args.arch == 'resnet50_cifar10':
		model = ResNet50()
	elif args.arch == "resnet50_imagenet":
		model = models.__dict__['resnet50']()
	elif args.arch == "mobilenet_v2_cifar10":
		model = MobileNetV2()
	## Move to GPU
	torch.cuda.empty_cache()
	model = model.to(args.gpu_prune)
	print("Using GPUs: {}".format(args.gpu_prune))
	criterion = nn.CrossEntropyLoss().cuda(args.gpu_prune)
	## LOAD STARTING CHECKPOINT
	if args.data_set == 'imagenet' and args.arch == 'resnet50_imagenet':
		args.resume = 'ResNet50_Imagenet_77.04_unwrapped.pth'
		pre_acc = 77.04
	elif args.data_set == 'cifar10' and args.arch == 'wrn-28-8':
		args.resume = 'WRN-28-8-Dense-New-New.pth.tar'
		pre_acc = 96.04
	elif args.data_set == 'mnist' and args.arch == 'lenet5caffe':
		args.resume = 'mnist_lenet5_99.08_r2.pt'
		pre_acc = 99.08
	elif args.data_set == 'cifar10' and args.arch.startswith("wrn"):
		args.resume = 'WRN-22-8_94.0.pt'
		pre_acc = 94.0
	elif args.data_set == 'cifar10' and args.arch == 'resnet50_cifar10':
		args.resume = 'ResNet50_CIFAR10_95.16_unwrapped.pth'
		pre_acc = 95.16	
	elif args.data_set == 'cifar10' and args.arch == 'mobilenet_v2_cifar10':
		pre_acc = float(args.resume[-8:-3])	
	if os.path.isfile(args.resume):
		checkpoint = torch.load(args.resume, map_location='cuda:0')
		model.load_state_dict(checkpoint)
		print("=> loaded checkpoint '{}'".format(args.resume))
	else:
		print("=> no checkpoint found at '{}'".format(args.resume))
	
	threshold = genfromtxt('./layer_sparsities.csv', delimiter=',') 

	columns = ['Layer Index','Layer Weights','Pre Accuracy','Weights Pruned','% Pruning','Post Accuracy']
	df = pd.DataFrame(columns=columns)
	layer_index=0
	mask_arr = []
	log_path = "{}/prune_refined_log.csv".format(args.log_dir)
	for child in list(model.modules()):
		if (isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)):
			# Do pruning
			num_pruned, num_weights, mask = prune_with_percent(child, threshold[layer_index])
			mask_arr.append(mask)
			pctpruned = 100*num_pruned/num_weights
			print("Pruned layer: {} Pct pruned: {}".format(layer_index, pctpruned))
			layer_index += 1
			#Log results
			df2 = pd.DataFrame([[layer_index, num_weights, pre_acc, num_pruned, pctpruned, "NA"]], columns=columns)
			df = df.append(df2, ignore_index=True)
			df.to_csv(log_path)
	#Save pruned model and weight mask
	filename='./prunedmodel.pth.tar'
	state={'state_dict': model.state_dict()}
	torch.save(state, filename)
	np.save("./weight_mask", mask_arr)
	#Save log
	post_acc = float(validate_pruning(val_loader, model, criterion, args))
	df2 = pd.DataFrame([["", "", "", "", "", post_acc]], columns=columns)
	df = df.append(df2, ignore_index=True)
	df.to_csv(log_path)

def prune_with_percent(m, threshold):
	"""
	Prunes a given layer and creates a weightmask for this layer
	Args:
		m: layer to prune
		threshold: % pruning to be done
	"""
	num_weights = torch.numel(m.weight.data)
	x = torch.flatten(m.weight.data.abs())
	k = int (threshold * num_weights)
	if k==0: k=1
	weight_threshold = torch.kthvalue(x,k).values
	weight_mask = torch.ge(m.weight.data.abs(), weight_threshold).type('torch.cuda.FloatTensor')
	m.weight.data *= weight_mask
	num_pruned = num_weights - torch.nonzero(weight_mask).size(0)
	return num_pruned, num_weights, weight_mask

def main():
	args = parser.parse_args()
	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])
	args.distributed = args.world_size > 1 or args.multiprocessing_distributed
	ngpus_per_node = torch.cuda.device_count()
	print("Executed the main function")
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
		# mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, weight_mask_arr))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)
def main_worker(gpu, ngpus_per_node, args):
	# if gpu==0:
	# 	wandb.init(project="GlobalPruning-Exp3.4")
	# 	wandb.run.name = "Epochs="+str(args.epochs)+",LR="+str(args.lr)+",Step="+str(args.step_size)+",Gamma="+str(args.gamma)+",Batch="+str(args.batch_size)
	# 	wandb.run.save()
	global best_acc1
	args.gpu = gpu
	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))
	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
	# create model
	print("=> creating model '{}'".format(args.arch))
	if args.arch.startswith("efficientnet"):
		model = EfficientNet.from_name(args.arch)
	elif args.arch == "mobilenet_v2":
		model = models.__dict__[args.arch]()
	elif args.arch.startswith("wrn"):
		cls, cls_args = wrn_models[args.arch]
		model = cls(*(cls_args))
		args.nesterov = True
	elif args.arch == 'lenet5caffe':
		model = LeNet5Caffe()
	elif args.arch == 'resnet50_cifar10':
		model = ResNet50()
	elif args.arch == "resnet50_imagenet":
		model = models.__dict__['resnet50']()
	elif args.arch == "mobilenet_v2_cifar10":
		model = MobileNetV2()
	# Load original pruned model if starting fresh fine-tuning cycle
	if not args.resume:
		if os.path.isfile(args.model):
			print("=> loading model '{}'".format(args.model))
			checkpoint = torch.load(args.model, map_location='cuda:0')
			model.load_state_dict(checkpoint['state_dict'])
			args.start_epoch = 0
			if args.data_set == 'imagenet' and args.arch == 'resnet50_imagenet':
				best_acc1 = 0.1
				starting_acc = 77.04
				sparsity = 90
			elif args.data_set == 'mnist' and args.arch == 'lenet5caffe':
				best_acc1 = 46.5
				starting_acc = 99.10
				sparsity = 99
			elif args.data_set == 'cifar10' and args.arch.startswith("wrn"):
				best_acc1 = 10
				starting_acc = 94.0
				sparsity = 95	
			elif args.data_set == 'cifar10' and args.arch.startswith("mobile"):
				best_acc1 = 10
				starting_acc = 94.13
				sparsity = 40	
			print("=> loaded checkpoint '{}' (epoch {})".format(args.model, args.start_epoch))
		else:
			print("=> no model found at '{}'".format(args.model))
	'''Supporting multiprocessing distributed scheme only for training. Not supporting other schemes'''
	# For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise,
	# DistributedDataParallel will use all available devices.
	torch.cuda.set_device(args.gpu)
	model.cuda(args.gpu)
	# When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size
	# ourselves based on the total number of GPUs we have
	args.batch_size = int(args.batch_size / ngpus_per_node)
	args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
	# define loss function (criterion) and optimizer
	criterion = LabelSmoothing(args.smoothing)
	print("Nesterov is ", args.nesterov)
	optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume, map_location='cuda:0')
			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			if args.gpu is not None:
				# best_acc1 may be from a checkpoint from a different GPU
				best_acc1 = best_acc1.to(args.gpu)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	cudnn.benchmark = True
	# Data loading code
	
	if args.data_set == 'imagenet':
		traindir = '/datasets/imagenet/train'
		valdir = '/datasets/imagenet/val'
		train_dataset = datasets.ImageFolder(traindir,
				transforms.Compose([
					transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				]))
		val_dataset = datasets.ImageFolder(valdir, 
				transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				]))
	elif args.data_set == 'mnist':
		data_path = '/data/'
		train_dataset = datasets.MNIST(data_path, train=True, download=True,
						transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.1307,), (0.3081,))
                    	]))
		val_dataset = datasets.MNIST(data_path, train=False, 
						transform=transforms.Compose([
							transforms.ToTensor(), 
							transforms.Normalize((0.1307,), (0.3081,))
						]))
	elif args.data_set == 'cifar10':
		normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		if args.arch.startswith("mobile"):
			print("Lambda is off")
			train_transform = transforms.Compose([
				transforms.RandomCrop(32),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
				])
		else:
			print("Lambda is on")
			train_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Lambda(transform_func),
				transforms.ToPILImage(),
				transforms.RandomCrop(32),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
				])
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			normalize
			])
		train_dataset = datasets.CIFAR10('/data/', train=True, transform=train_transform, download=False)
		val_dataset = datasets.CIFAR10('/data/', train=False, transform=test_transform, download=False)	

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	else:
		train_sampler = None
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
		num_workers=args.workers, pin_memory=True, sampler=train_sampler)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

	if args.evaluate:
		validate_finetuning(val_loader, model, criterion, args)
		return
	np_load_old = np.load
	# modify the default parameters of np.load
	np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
	# call np.load with allow_pickle implicitly set to true
	if args.mask:
		path = args.mask
	weight_mask = np.load(path)
	print("Using weights mask: {}".format(path))
	# restore np.load for future normal usage
	np.load = np_load_old
	#Multiprocessing notes - To share tensor of tensors, add the tensors to a list. Using numpy as a container will fail. Python lists will work
	current_device = torch.cuda.current_device()
	print("Current device is ", current_device)
	weight_mask_arr = []
	for item in weight_mask:
		weight_mask_arr.append(item.cuda())
	columns = ['Epoch','Current Time','LR','Orig Acc','% Pruning','Train Acc','Test Acc','Best Test Acc']
	df = pd.DataFrame(columns=columns)
	log_path = "./Run6_log_Epochs="+str(args.epochs)+",LR="+str(args.lr)+",Batch="+str(args.batch_size)+".csv"
	for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)
		lr = adjust_learning_rate(optimizer, epoch, args)
		# train_acc = acc1 = torch.tensor(1)
		# train for one epoch
		train_acc = train(train_loader, model, criterion, optimizer, epoch, args, weight_mask_arr)
		# if gpu==0:
		# 	wandb.log({"Train Accuracy": train_acc.item()})  
		# evaluate on validation set
		acc1 = validate_finetuning(val_loader, model, criterion, args)
		# if gpu==0:
		# 	wandb.log({"Test Accuracy": acc1.item()})
		# remember best acc@1 and save checkpoint
		is_best = acc1 > best_acc1
		best_acc1 = max(acc1, best_acc1)
		time = datetime.datetime.now()
		df2 = pd.DataFrame([[epoch, time, lr, starting_acc, sparsity, train_acc.item(), acc1.item(), best_acc1.item()]], columns=columns)
		df = df.append(df2, ignore_index=True)
		df.to_csv(log_path)
		if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
			save_checkpoint({
				'epoch': epoch,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'best_acc1': best_acc1,
				'optimizer' : optimizer.state_dict(),
			}, is_best, args)
		torch.cuda.empty_cache()

def train(train_loader, model, criterion, optimizer, epoch, args, weight_mask_arr):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
							top5, prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		if args.gpu is not None:
			input = input.cuda(args.gpu, non_blocking=True)	

		target = target.cuda(args.gpu, non_blocking=True)
		# compute output
		output = model(input)
		loss = criterion(output, target)

		# measure accuracy and record loss
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), input.size(0))
		top1.update(acc1[0], input.size(0))
		top5.update(acc5[0], input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()		
		optimizer.step()

		#Flush weights
		id = 0
		for child in list(model.modules()):
			if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
				child.weight.data *= weight_mask_arr[id]
				# print("Weight size ", child.weight.data.size())
				# print("Mask size ", weight_mask_arr[id].size())
				# print("GPU is {} Layer is {} Mask sum is {}" .format(torch.cuda.current_device(), id, weight_mask_arr[id].data.sum()))
				# print("GPU is {} Layer is {} Mask sum is {} Non-Zero weights are {}" .format(torch.cuda.current_device(), id, weight_mask_arr[id].data.sum(), torch.nonzero(child.weight.data).size(0)))
				id+=1

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			progress.print(i)

	print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
	return top1.avg
def validate_finetuning(val_loader, model, criterion, args):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
							prefix='Test: ')

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(val_loader):
			if args.gpu is not None:
				input = input.cuda(args.gpu, non_blocking=True)
			target = target.cuda(args.gpu, non_blocking=True)

			# compute output
			output = model(input)
			loss = criterion(output, target)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), input.size(0))
			top1.update(acc1[0], input.size(0))
			top5.update(acc5[0], input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				progress.print(i)

		# TODO: this should also be done with the ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
	return top1.avg
def validate_pruning(val_loader, model, criterion, args):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
							prefix='Test: ')

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(val_loader):
			if args.gpu_prune is not None:
				input = input.cuda(args.gpu_prune, non_blocking=True)
			target = target.cuda(args.gpu_prune, non_blocking=True)

			# compute output
			output = model(input)
			loss = criterion(output, target)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), input.size(0))
			top1.update(acc1[0], input.size(0))
			top5.update(acc5[0], input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				progress.print(i)

		# TODO: this should also be done with the ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
	return top1.avg
def save_checkpoint(state, is_best, args):
	if is_best:
		print("New best model saved")
		best_model_name = "Run6_BestModel_Epoch{}.pth.tar".format(state['epoch'])
		torch.save(state, best_model_name)
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
	def __init__(self, num_batches, *meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def print(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'
def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / args.warmup_epochs
    else:
        e = epoch - args.warmup_epochs
        es = args.epochs - args.warmup_epochs
        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr
    print("Epoch is {} LR is {}".format(epoch,lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res
def transform_func(x):
	return F.pad(x.unsqueeze(0), (4,4,4,4),mode='reflect').squeeze()
if __name__ == '__main__':
	min_weight_target, val_loader = step()
	step2()
	step3()
	main()
