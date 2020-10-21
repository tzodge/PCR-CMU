#!/usr/bin/env python
# -*- coding: utf-8 -*-

# example usage:
# python main.py --exp_name=num_points_512_heads_8_exp2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd  --batch_size=16 --factor=1 --num_points=512  --lr=0.005 --n_heads=8 --eval

from __future__ import print_function
import open3d as o3d 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import open3d as o3d 
import pdb
import os
import gc
import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from data import threedmatch
from model import DCP
from util import transform_point_cloud, npmat2euler,error_euler_angles
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch import autograd
import helper
import pdb 

# Part of the code is referred from: https://github.com/floodsung/LearningToCompare_FSL


epoch_COUNT = 0

class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()


def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
	os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
	os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
	os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')



def test_one_epoch(args, net, test_loader):
	net.eval()

	# initialization
	mse_ab = 0
	mae_ab = 0
	mse_ba = 0
	mae_ba = 0

	total_loss = 0
	total_cycle_loss = 0
	num_examples = 0
	rotations_ab = []
	translations_ab = []
	rotations_ab_pred = []
	translations_ab_pred = []


	rotations_ba = []
	translations_ba = []
	rotations_ba_pred = []
	translations_ba_pred = []

	eulers_ab = []
	eulers_ba = []

	batch_idx=0
	total_correct_pred = 0
	itr = 0
	if args.debug:
		ang_error_list = []
	for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, col_idx, corr_mat_ab in tqdm(test_loader):
		batch_size = src.size(0)
		num_points = src.size(-1)
		num_points_target = target.size(-1)
		
		if args.debug: # if degubbing
			for i in range(batch_size):
				np.savetxt("variables_storage/src_batch_{}_sample_{}".format(batch_idx,i), src[i,:,:])
				np.savetxt("variables_storage/target_batch_{}_sample_{}".format(batch_idx,i), target[i,:,:])
				np.savetxt("variables_storage/rotation_ab_batch_{}_sample_{}".format(batch_idx,i),rotation_ab[i,:,:])
				np.savetxt("variables_storage/translation_ab_batch_{}_sample_{}".format(batch_idx,i),translation_ab[i,:])
				np.savetxt("variables_storage/euler_ab_batch_{}_sample_{}".format(batch_idx,i),euler_ab[i,:])
				np.savetxt("variables_storage/col_idx_batch_{}_sample_{}".format(batch_idx,i),col_idx[i,:])
			if batch_idx >= 100 :
				break                   

		src = src.cuda()
		target = target.cuda()
		rotation_ab = rotation_ab.cuda()
		translation_ab = translation_ab.cuda()
		rotation_ba = rotation_ba.cuda()
		translation_ba = translation_ba.cuda()
		col_idx = col_idx.cuda()
		corr_mat_ab = corr_mat_ab.cuda()

		num_examples += batch_size

		# model output
		rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred, corr_mat_ab_pred = net(src, target)

		## save rotation and translation
		rotations_ab.append(rotation_ab.detach().cpu().numpy())
		translations_ab.append(translation_ab.detach().cpu().numpy())
		rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
		translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
		eulers_ab.append(euler_ab.numpy())
		
		rotations_ba.append(rotation_ba.detach().cpu().numpy())
		translations_ba.append(translation_ba.detach().cpu().numpy())
		rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
		translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
		eulers_ba.append(euler_ba.numpy())

		# transforming the point cloud according to given rotation and translation
		transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

		transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

		identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
		
 
 		#  correspondence loss, as proposed in our paper
		if args.loss == 'cross_entropy_corr':
			# corr_mat_ab: ground truth correspondence matrix
			# corr_mat_ab_pred: predicted correspondence matrix
			loss_corr = F.cross_entropy(corr_mat_ab_pred.view(batch_size*num_points,num_points_target), 
								   torch.argmax(corr_mat_ab.transpose(1,2).reshape(-1,num_points_target),axis =1)) 

			loss_transf = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
				   + F.mse_loss(translation_ab_pred, translation_ab)

			loss = loss_corr


		# translation loss, as proposed in DCP
		elif args.loss == 'mse_transf':
			loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
			   + F.mse_loss(translation_ab_pred, translation_ab)) 
		else:
			raise Exception ("please verify the input loss function")


		if args.cycle:
			raise Exception ("cycle for corr_mat_ab not implemented yet")
			rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
			translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
														translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
										   + translation_ba_pred) ** 2, dim=[0, 1])
			cycle_loss = rotation_loss + translation_loss

			loss = loss + cycle_loss * 0.1

		total_loss += loss.item() * batch_size

		if args.cycle:
			total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

 

		gt_idx = torch.argmax(corr_mat_ab.transpose(1,2).reshape(-1,num_points_target),axis =1) # ground-truth index of the corresponding target point 
		pred_idx = torch.argmax(corr_mat_ab_pred.view(-1,num_points_target),axis =1) # predicted index of the corresponding target point 

		# if the indices match, then the predicted corresponding target point is correct
		correct_pred_idx=torch.where(gt_idx-pred_idx ==0) 
		total_correct_pred += len(correct_pred_idx[0])


		mse_ab +=  torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
		mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

		mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
		mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size



		if args.debug:            
			corr_mat_ab_pred_np = torch.clone(corr_mat_ab_pred).detach().cpu().numpy()         
			corr_mat_ab_gt_np = torch.clone(corr_mat_ab).detach().cpu().numpy()         
			rotation_ab_pred_np = torch.clone(rotation_ab_pred).detach().cpu().numpy()         
			translation_ab_pred_np = torch.clone(translation_ab_pred).detach().cpu().numpy()         
			col_idx_pred = np.argmax(corr_mat_ab_pred_np,axis=1) 
			
			for i in range(batch_size):
				np.savetxt("variables_storage/corr_mat_ab_pred_batch_{}_sample_{}".format(batch_idx,i),corr_mat_ab_pred_np[i,:,:])
				np.savetxt("variables_storage/col_idx_pred_batch_{}_sample_{}".format(batch_idx,i),col_idx_pred[i,:])
				np.savetxt("variables_storage/rotation_ab_pred_batch_{}_sample_{}".format(batch_idx,i),rotation_ab_pred_np[i,:])
				np.savetxt("variables_storage/corr_mat_ab_gt_{}_sample_{}".format(batch_idx,i),corr_mat_ab_gt_np[i,:,:])
			

		itr+=1  
		batch_idx +=1

	# computing percentage of incorrect point correspondences
	incorrect_correspondences =   (1 - total_correct_pred/(num_examples*num_points))*100
	print ( incorrect_correspondences," percent incorrect testing predictions ")

	rotations_ab = np.concatenate(rotations_ab, axis=0)
	translations_ab = np.concatenate(translations_ab, axis=0)
	rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
	translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

	rotations_ba = np.concatenate(rotations_ba, axis=0)
	translations_ba = np.concatenate(translations_ba, axis=0)
	rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
	translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

	eulers_ab = np.concatenate(eulers_ab, axis=0)
	eulers_ba = np.concatenate(eulers_ba, axis=0)

	return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
		   mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
		   mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
		   translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
		   translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba,\
		   incorrect_correspondences


def train_one_epoch(args, net, train_loader, opt):
	net.train()
	global epoch_COUNT
	mse_ab = 0
	mae_ab = 0
	mse_ba = 0
	mae_ba = 0

	total_loss = 0
	total_loss_dcp_rot = 0
	total_loss_dcp_t = 0
	total_cycle_loss = 0
	num_examples = 0
	rotations_ab = []
	translations_ab = []
	rotations_ab_pred = []
	translations_ab_pred = []

	rotations_ba = []
	translations_ba = []
	rotations_ba_pred = []
	translations_ba_pred = []

	eulers_ab = []
	eulers_ba = []

	total_correct_pred =0
	itr = 0
	for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, col_idx, corr_mat_ab in tqdm(train_loader):
		src = src.cuda()
		target = target.cuda()
		rotation_ab = rotation_ab.cuda()
		translation_ab = translation_ab.cuda()
		rotation_ba = rotation_ba.cuda()
		translation_ba = translation_ba.cuda()
		col_idx = col_idx.cuda()
		corr_mat_ab = corr_mat_ab.cuda()

		batch_size = src.size(0)
		num_points = src.size(-1)
		num_points_target = target.size(-1)

		opt.zero_grad()
		num_examples += batch_size

		# model output
		rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred, corr_mat_ab_pred = net(src, target) 

		## save rotation and translation
		rotations_ab.append(rotation_ab.detach().cpu().numpy())
		translations_ab.append(translation_ab.detach().cpu().numpy())
		rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
		translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
		eulers_ab.append(euler_ab.numpy())
		
		rotations_ba.append(rotation_ba.detach().cpu().numpy())
		translations_ba.append(translation_ba.detach().cpu().numpy())
		rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
		translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
		eulers_ba.append(euler_ba.numpy())

		# transforming the point cloud according to given rotation and translation
		transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

		transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

		identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)


		loss_dcp_rot = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) 
		loss_dcp_t = F.mse_loss(translation_ab_pred, translation_ab) 
				

 		#  correspondence loss, as proposed in our paper
		if args.loss == 'cross_entropy_corr':
			loss_corr = F.cross_entropy(corr_mat_ab_pred.view(batch_size*num_points,num_points_target), 
								   torch.argmax(corr_mat_ab.transpose(1,2).reshape(-1,num_points_target),axis =1)) 

			loss_transf = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
				   + F.mse_loss(translation_ab_pred, translation_ab)

			loss = loss_corr

		# translation loss, as proposed in DCP
		elif args.loss == 'mse_transf':
			loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
			   + F.mse_loss(translation_ab_pred, translation_ab))   
		else:
			raise Exception ("please verify the input loss function")



		if args.cycle:
			raise Exception ("cycle for corr_mat_ab not implemented yet")
			rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
			translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
														translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
										   + translation_ba_pred) ** 2, dim=[0, 1])
			cycle_loss = rotation_loss + translation_loss

			loss = loss + cycle_loss * 0.1
		 
		loss.backward()
		opt.step()
		total_loss += loss.item() * batch_size
		total_loss_dcp_rot += loss_dcp_rot.item() * batch_size
		total_loss_dcp_t += loss_dcp_t.item() * batch_size


		gt_idx = torch.argmax(corr_mat_ab.transpose(1,2).reshape(-1,num_points_target),axis =1) # ground-truth index of the corresponding target point 
		pred_idx = torch.argmax(corr_mat_ab_pred.view(-1,num_points_target),axis =1) # predicted index of the corresponding target point 

		# if the indices match, then the predicted corresponding target point is correct
		correct_pred_idx=torch.where(gt_idx-pred_idx ==0)
		total_correct_pred += len(correct_pred_idx[0])
		
		if args.cycle:
			total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

		mse_ab +=  torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
		mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

		mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
		mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size


		itr+=1  
       
	# computing percentage of incorrect point correspondences
	incorrect_correspondences =   (1 - total_correct_pred/(num_examples*num_points))*100
	print ( incorrect_correspondences," percent incorrect training predictions ")
	print ( loss_dcp_rot," loss_dcp_rot ")
	print ( loss_dcp_t," loss_dcp_t ")


	rotations_ab = np.concatenate(rotations_ab, axis=0)
	translations_ab = np.concatenate(translations_ab, axis=0)
	rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
	translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

	rotations_ba = np.concatenate(rotations_ba, axis=0)
	translations_ba = np.concatenate(translations_ba, axis=0)
	rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
	translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

	eulers_ab = np.concatenate(eulers_ab, axis=0)
	eulers_ba = np.concatenate(eulers_ba, axis=0)

	return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
		   mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
		   mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
		   translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
		   translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba,\
		   incorrect_correspondences


def test(args, net, test_loader, boardio, textio):

	test_loss, test_cycle_loss, \
	test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
	test_rotations_ab_pred, \
	test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
	test_translations_ba_pred, test_eulers_ab, test_eulers_ba, test_incorrect_correspondeces = test_one_epoch(args, net, test_loader)
	test_rmse_ab = np.sqrt(test_mse_ab)
	test_rmse_ba = np.sqrt(test_mse_ba)

	test_error_euler_angles_ab = error_euler_angles(test_rotations_ab_pred,np.degrees(test_eulers_ab))  # computing euler angle error
	test_r_mse_ab = np.mean(test_error_euler_angles_ab ** 2)
	test_r_rmse_ab = np.sqrt(test_r_mse_ab)
	test_r_mae_ab = np.mean(np.abs(test_error_euler_angles_ab))
	test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
	test_t_rmse_ab = np.sqrt(test_t_mse_ab)
	test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

	textio.cprint('==FINAL TEST==')
	textio.cprint('A--------->B')
	textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
				  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
				  % (-1, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab,
					 test_r_mse_ab, test_r_rmse_ab,
					 test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))


def train(args, net, train_loader, test_loader, boardio, textio):
	if args.use_sgd:
		print("Use SGD")
		opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
	else:
		print("Use Adam")
		opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4, betas=args.betas)
	scheduler = MultiStepLR(opt, milestones=[25, 50, 100, 150 ,200], gamma=0.25)

	print(len(train_loader),"len(train_loader)")
	best_test_loss = np.inf
	best_test_cycle_loss = np.inf
	best_test_mse_ab = np.inf
	best_test_rmse_ab = np.inf
	best_test_mae_ab = np.inf

	best_test_r_mse_ab = np.inf
	best_test_r_rmse_ab = np.inf
	best_test_r_mae_ab = np.inf
	best_test_t_mse_ab = np.inf
	best_test_t_rmse_ab = np.inf
	best_test_t_mae_ab = np.inf

	best_test_mse_ba = np.inf
	best_test_rmse_ba = np.inf
	best_test_mae_ba = np.inf

	best_test_r_mse_ba = np.inf
	best_test_r_rmse_ba = np.inf
	best_test_r_mae_ba = np.inf
	best_test_t_mse_ba = np.inf
	best_test_t_rmse_ba = np.inf
	best_test_t_mae_ba = np.inf

	for epoch in range(args.epochs):
		scheduler.step()
		train_loss, train_cycle_loss, \
		train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
		train_rotations_ab_pred, \
		train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
		train_translations_ba_pred, train_eulers_ab, train_eulers_ba, train_incorrect_correspondeces = train_one_epoch(args, net, train_loader, opt)

		test_loss, test_cycle_loss, \
		test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
		test_rotations_ab_pred, \
		test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
		test_translations_ba_pred, test_eulers_ab, test_eulers_ba, test_incorrect_correspondeces = test_one_epoch(args, net, test_loader)
		train_rmse_ab = np.sqrt(train_mse_ab)
		test_rmse_ab = np.sqrt(test_mse_ab)

		train_rmse_ba = np.sqrt(train_mse_ba)
		test_rmse_ba = np.sqrt(test_mse_ba)

		train_error_euler_angles_ab = error_euler_angles(train_rotations_ab_pred,np.degrees(train_eulers_ab))  # computing euler angle error
		train_r_mse_ab = np.mean(train_error_euler_angles_ab ** 2)
		train_r_rmse_ab = np.sqrt(train_r_mse_ab)
		train_r_mae_ab = np.mean(np.abs(train_error_euler_angles_ab))
		train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
		train_t_rmse_ab = np.sqrt(train_t_mse_ab)
		train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))


		test_error_euler_angles_ab = error_euler_angles(test_rotations_ab_pred,np.degrees(test_eulers_ab))  # computing euler angle error
		test_r_mse_ab = np.mean(test_error_euler_angles_ab ** 2)
		test_r_rmse_ab = np.sqrt(test_r_mse_ab)
		test_r_mae_ab = np.mean(np.abs(test_error_euler_angles_ab))
		test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
		test_t_rmse_ab = np.sqrt(test_t_mse_ab)
		test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

		if best_test_loss >= test_loss:
			best_test_loss = test_loss
			best_test_cycle_loss = test_cycle_loss

			best_test_mse_ab = test_mse_ab
			best_test_rmse_ab = test_rmse_ab
			best_test_mae_ab = test_mae_ab

			best_test_r_mse_ab = test_r_mse_ab
			best_test_r_rmse_ab = test_r_rmse_ab
			best_test_r_mae_ab = test_r_mae_ab

			best_test_t_mse_ab = test_t_mse_ab
			best_test_t_rmse_ab = test_t_rmse_ab
			best_test_t_mae_ab = test_t_mae_ab

			if torch.cuda.device_count() > 1:
				torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
			else:
				torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

		textio.cprint('==TRAIN==')
		textio.cprint('A--------->B')
		textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss:, %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
					  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f, percent_incorrect_corr: %f'
					  % (epoch, train_loss, train_cycle_loss, train_mse_ab, train_rmse_ab, train_mae_ab, train_r_mse_ab,
						 train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab, train_incorrect_correspondeces))

		textio.cprint('==TEST==')
		textio.cprint('A--------->B')
		textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
					  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f, percent_incorrect_corr: %f' 
					  % (epoch, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab, test_r_mse_ab,
						 test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab, test_incorrect_correspondeces))

		textio.cprint('==BEST TEST==')
		textio.cprint('A--------->B')
		textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
					  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
					  % (epoch, best_test_loss, best_test_cycle_loss, best_test_mse_ab, best_test_rmse_ab,
						 best_test_mae_ab, best_test_r_mse_ab, best_test_r_rmse_ab,
						 best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab))

		# Train
		boardio.add_scalar('A->B/train/loss', train_loss, epoch)
		boardio.add_scalar('A->B/train/MSE', train_mse_ab, epoch)
		boardio.add_scalar('A->B/train/RMSE', train_rmse_ab, epoch)
		boardio.add_scalar('A->B/train/MAE', train_mae_ab, epoch)
		boardio.add_scalar('A->B/train/rotation/MSE', train_r_mse_ab, epoch)
		boardio.add_scalar('A->B/train/rotation/RMSE', train_r_rmse_ab, epoch)
		boardio.add_scalar('A->B/train/rotation/MAE', train_r_mae_ab, epoch)
		boardio.add_scalar('A->B/train/translation/MSE', train_t_mse_ab, epoch)
		boardio.add_scalar('A->B/train/translation/RMSE', train_t_rmse_ab, epoch)
		boardio.add_scalar('A->B/train/translation/MAE', train_t_mae_ab, epoch)
		boardio.add_scalar('A->B/train/incorrect_correspondences', train_incorrect_correspondeces, epoch)


		# Test
		boardio.add_scalar('A->B/test/loss', test_loss, epoch)
		boardio.add_scalar('A->B/test/MSE', test_mse_ab, epoch)
		boardio.add_scalar('A->B/test/RMSE', test_rmse_ab, epoch)
		boardio.add_scalar('A->B/test/MAE', test_mae_ab, epoch)
		boardio.add_scalar('A->B/test/rotation/MSE', test_r_mse_ab, epoch)
		boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
		boardio.add_scalar('A->B/test/rotation/MAE', test_r_mae_ab, epoch)
		boardio.add_scalar('A->B/test/translation/MSE', test_t_mse_ab, epoch)
		boardio.add_scalar('A->B/test/translation/RMSE', test_t_rmse_ab, epoch)
		boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)
		boardio.add_scalar('A->B/test/incorrect_correspondences', test_incorrect_correspondeces, epoch)

		# Best Test
		boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
		boardio.add_scalar('A->B/best_test/MSE', best_test_mse_ab, epoch)
		boardio.add_scalar('A->B/best_test/RMSE', best_test_rmse_ab, epoch)
		boardio.add_scalar('A->B/best_test/MAE', best_test_mae_ab, epoch)
		boardio.add_scalar('A->B/best_test/rotation/MSE', best_test_r_mse_ab, epoch)
		boardio.add_scalar('A->B/best_test/rotation/RMSE', best_test_r_rmse_ab, epoch)
		boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
		boardio.add_scalar('A->B/best_test/translation/MSE', best_test_t_mse_ab, epoch)
		boardio.add_scalar('A->B/best_test/translation/RMSE', best_test_t_rmse_ab, epoch)
		boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)

		if torch.cuda.device_count() > 1:
			torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%s.t7' % (args.exp_name, "last"))
		else:
			torch.save(net.state_dict(), 'checkpoints/%s/models/model.%s.t7' % (args.exp_name, "last"))
		gc.collect()




# functions for testing on a single point cloud: test_bunny
def extract_vertices(mesh,num_vert=512):
	pcd_downsampled = mesh.sample_points_uniformly( number_of_points=num_vert)  
	points = np.asanyarray(pcd_downsampled.points)  
	return points


def rotate_cloud (points, add_noise = False):
	'''
	input:
		points = Nx3
	output:
		points = Nx3
	'''
	axis = np.random.rand(3,) - np.array([0.5,0.5,0.5])  
	axis = axis/np.linalg.norm(axis)
	# angle from [-pi/factor to +pi/factor]
	angle = 2*(np.random.uniform()-0.5) * np.pi / 1
	 
	Rot_instance = Rotation.from_rotvec(axis*angle) 
	R_ab = Rot_instance.as_dcm()

	points_rot = R_ab.dot(points.T).T
	
	
	if add_noise :
		points_rot = points_rot + np.random.rand(len(points_rot),3)*0.05
	shuffle_idx = np.random.permutation(np.arange(len(points_rot)))
	points_rot = points_rot[shuffle_idx,:]
	return points_rot,R_ab



def Network_input_format(points):
	points_cuda = points.reshape(512,3,1).astype('float32')
	points_cuda = torch.from_numpy(points_cuda)
	points_cuda = points_cuda.permute(2,1,0).cuda() # input shape shpuld be [1,3,512]
	return points_cuda

def create_pcd_obj(np_array,col=[1,0,0]):
	'''
	input: nx3 array
	output: pcd object 
			can be displayed using o3d.visualization.draw_geometries([pcd1,pcd2])
	'''

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(np_array[:,0:3])
	pcd.paint_uniform_color(col)
	return pcd


def test_bunny(args,net):

	pcd_src = o3d.io.read_point_cloud("CAD_models/office_1_chair_extracted1.pcd")
	points_src = np.asarray(pcd_src.points)
	points_src = helper.fit_in_m1_to_1(points_src)

	pcd_target = o3d.io.read_point_cloud("CAD_models/office_1_chair_extracted1.pcd")
	points_target = np.asarray(pcd_target.points)
	points_target = helper.fit_in_m1_to_1(points_target)

	points_rot,R_ab_gt = rotate_cloud(points_target)
	
	points_cuda = Network_input_format(points_src)
	points_rot_cuda = Network_input_format(points_rot)

	src = points_cuda  #
	target = points_rot_cuda  #
 

	rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred, corr_mat_ab_pred = net(src, target) 
	print(rotation_ab_pred,"rotation_ab_pred")
	print(R_ab_gt,"R_ab_gt")


	points_pred = rotation_ab_pred.detach().cpu().numpy().dot(points_src.T).T

	helper.display_three_clouds(points_src,points_rot,points_pred,title="real world data" ,\
								legend_list=[" source","target","prediction"]	)


# main
def main():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--model', type=str, default='dcp', metavar='N',
						choices=['dcp'],
						help='Model to use, [dcp]')
	parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
						choices=['pointnet', 'dgcnn'],
						help='Embedding nn to use, [pointnet, dgcnn]')
	parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
						choices=['identity', 'transformer'],
						help='Attention-based pointer generator to use, [identity, transformer]')
	parser.add_argument('--head', type=str, default='svd', metavar='N',
						choices=['mlp', 'svd', ],
						help='Head to use, [mlp, svd]')
	parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
						help='Dimension of embeddings')
	parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
						help='Num of blocks of encoder&decoder')
	parser.add_argument('--n_heads', type=int, default=4, metavar='N',
						help='Num of heads in multiheadedattention')
	parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
						help='Num of dimensions of fc in transformer')
	parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
						help='Dropout ratio in transformer')
	parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
						help='Size of batch)')
	parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
						help='Size of batch)')
	parser.add_argument('--epochs', type=int, default=250, metavar='N',
						help='number of episode to train ')
	parser.add_argument('--use_sgd', action='store_true', default=False,
						help='Use SGD')
	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default: 0.001, 0.1 if using sgd)')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
						help='SGD momentum (default: 0.9)')
	parser.add_argument('--no_cuda', action='store_true', default=False,
						help='enables CUDA training')
	parser.add_argument('--seed', type=int, default=1234, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--eval', action='store_true', default=False,
						help='evaluate the model')
	parser.add_argument('--cycle', type=bool, default=False, metavar='N',
						help='Whether to use cycle consistency')
	parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
						help='Wheter to add gaussian noise')
	parser.add_argument('--unseen', type=bool, default=False, metavar='N',
						help='Wheter to test on unseen category')
	parser.add_argument('--num_points', type=int, default=1024, metavar='N',
						help='Num of points to use')
	parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40','threedmatch'], metavar='N',
						help='dataset to use')
	parser.add_argument('--factor', type=float, default=4, metavar='N',
						help='Divided factor for rotations')
	parser.add_argument('--model_path', type=str, default='', metavar='N',
						help='Pretrained model path')
	parser.add_argument('--betas', type=float, default=(0.9,0.999), metavar='N', nargs='+',
						help='Betas for adam')
	parser.add_argument('--same_pointclouds', type=bool, default=False, metavar='N',
						help='R*src + t should be exactly same as target')
	parser.add_argument('--debug', type=bool, default=False, metavar='N',
						help='saves variables in folder variables_storage')
	parser.add_argument('--num_itr_test', type=int, default=1, metavar='N',
						help='Num of net() during testing')
	parser.add_argument('--loss', type=str, default='cross_entropy_corr', metavar='N',
						choices=['cross_entropy_corr','mse_transf'],
						help='loss function: choose one of [mse_transf or cross_entropy_corr]')
	parser.add_argument('--cut_plane', type=bool, default=False, metavar='N',
						help='generates partial data')
	parser.add_argument('--one_cloud', type=bool, default=False, metavar='N',
						help='test for one unseen cloud')
	parser.add_argument('--partial', type=float, default = 0.0, metavar='N',
						help='partial = 0.1 ==> (num_points*partial) will be removed')
	parser.add_argument('--pretrained', type=bool, default = False, metavar='N',
						help='load pretrained model')	

	args = parser.parse_args()

	# for deterministic training
	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
	_init_(args)

	textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
	textio.cprint(str(args))

	# dataloading
	num_workers = 32
	if args.dataset == 'modelnet40':
		train_dataset = ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
					   unseen=args.unseen, factor=args.factor, same_pointclouds=args.same_pointclouds,
					   partial=args.partial,cut_plane=args.cut_plane) 
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,num_workers=num_workers)

		test_dataset = ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
					   unseen=args.unseen, factor=args.factor, same_pointclouds=args.same_pointclouds,
					   partial=args.partial,cut_plane=args.cut_plane)
		test_loader = DataLoader( test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False,num_workers=num_workers)

	else:
		raise Exception("not implemented")

	# model loading
	if args.model == 'dcp':
		net = DCP(args).cuda()
		if args.eval:
			if args.model_path is '':
				model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
			else:
				model_path = args.model_path
				print(model_path)
			if not os.path.exists(model_path):
				print("can't find pretrained model")
				return
			net.load_state_dict(torch.load(model_path), strict=False)
		if args.pretrained:
			if args.model_path =='':
				print ('Please specify path to pretrained weights \n For Ex: checkpoints/partial_global_512_identical/models/model.best.t7')
			else: 
				model_path = args.model_path
			print ("Using pretrained weights stored at:\n{}".format(model_path))
			net.load_state_dict(torch.load(model_path), strict=False)


		if torch.cuda.device_count() > 1:
			net = nn.DataParallel(net)
			print("Let's use", torch.cuda.device_count(), "GPUs!")
	else:
		raise Exception('Not implemented')
	
 
	# training and evaluation
	if args.eval:
		if args.one_cloud: # testing on a single point cloud
			print("one_cloud")
			test_bunny(args,net)

		else:
			test(args, net, test_loader, boardio, textio)

	else:
		train(args, net, train_loader, test_loader, boardio, textio)


	print('FINISH')
	boardio.close()


if __name__ == '__main__':
	main()
