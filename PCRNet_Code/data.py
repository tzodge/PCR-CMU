#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import open3d as o3d
import os
import sys
import glob
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from scipy.spatial import KDTree
try:
	import helper 
	import MinkowskiEngine as ME
	from helper import print_rounded
except:
	print("Some files are not imported.")	
import matplotlib.pyplot as plt
import transforms3d.axangles as t3d_axang

def uniform_2_sphere(num: int = None):
	"""Uniform sampling on a 2-sphere
	Source: https://gist.github.com/andrewbolster/10274979
	Args:
		num: Number of vectors to sample (or None if single)
	Returns:
		Random Vector (np.ndarray) of size (num, 3) with norm 1.
		If num is None returned value will have size (3,)
	"""
	if num is not None:
		phi = np.random.uniform(0.0, 2 * np.pi, num)
		cos_theta = np.random.uniform(-1.0, 1.0, num)
	else:
		phi = np.random.uniform(0.0, 2 * np.pi)
		cos_theta = np.random.uniform(-1.0, 1.0)

	theta = np.arccos(cos_theta)
	x = np.sin(theta) * np.cos(phi)
	y = np.sin(theta) * np.sin(phi)
	z = np.cos(theta)

	return np.stack((x, y, z), axis=-1)

def make_open3d_point_cloud(xyz, color=None):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	if color is not None:
		pcd.colors = o3d.utility.Vector3dVector(color)
	return pcd
# The code is referred from: https://github.com/WangYueFt/dcp 

def download():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, '../datasets')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = '--no-check-certificates https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))


def load_data(partition):
#     download()
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, '../datasets')       # change accordingly
	all_data = []
	all_label = []
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
		f = h5py.File(h5_name)
		data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label


# def translate_pointcloud(pointcloud):
#     xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
#     xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

#     translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
#     return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
	N, C = pointcloud.shape
	pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
	return pointcloud




class ModelNet40(Dataset):
	
	def __init__(self, 
				 num_points=1024,         # points to sample
				 gaussian_noise=False,    # N(0,0.01) clip=0.05               
				 partial=1.0,             # %points to keep in pointcloud1
				 identical=True,          # identical=same points sampled for both src and tgt
				 factor=1,                # rotation sampled from (-pi/factor, pi/factor)
				 trans_mag=0.5,           # translation sampled from uniform(-trans_mag,trans_mag) 
				 method='dcp_modified',   # choices:[dcp_modified, dcp, prnet, pointnetlk, pcrnet, pcrnet_corr, rpmnet]
				 partition='train',
				 unseen=False,
				 single_category=False): 
		

		self.data, self.label = load_data(partition)
		print (len(self.data),"len(self.data)")
		self.num_points = num_points
		self.partition = partition
		self.gaussian_noise = gaussian_noise
		self.identical = identical
		self.unseen = unseen
		self.label = self.label.squeeze()
		self.factor = factor
		self.trans_mag = trans_mag
		self.partial = partial
		self.num_points_CAD = len(self.data[0])
		self.single_category = single_category
		self.outl_thresh = 1
		self.method = method

		### Only one asymmetric category
		if self.single_category:
			category_id = 7 # car


			if self.partition == 'test':
				self.data = self.data[self.label==category_id]
				self.label = self.label[self.label==category_id]
			elif self.partition == 'train':                
				self.data = self.data[self.label==category_id]
				self.label = self.label[self.label==category_id]



		if self.unseen:
			print ("for unseen data")
			####### simulate testing on first 20 categories while training on last 20 categories
			if self.partition == 'test':
				self.data = self.data[self.label>=20]
				self.label = self.label[self.label>=20]
			elif self.partition == 'train':                
				self.data = self.data[self.label<20]
				self.label = self.label[self.label<20]


	def __getitem__(self, item):
		if self.partition != 'train':
			np.random.seed(item)
			
############################## GT Transformation Generation ##########################
		axis = uniform_2_sphere().flatten()   # uniform sampling from a sphere

		### angle from [-pi/factor to +pi/factor]
		angle = 2*(np.random.uniform()-0.5) * np.pi / self.factor
		Rot_instance = Rotation.from_rotvec(axis*angle) 
		R_ab = Rot_instance.as_dcm()

		[anglez,angley,anglex] = Rot_instance.as_euler('zyx') 

		R_ba = R_ab.T
		rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
		
		translation_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)
		translation_ba = -R_ba.dot(translation_ab)

		euler_ab = np.asarray([anglez, angley, anglex])
		euler_ba = -euler_ab[::-1]
		
########################################################################################
#                     Pointcloud Processing    
########################################################################################


		if self.identical:
			choose_idx = np.random.choice(range(self.num_points_CAD),self.num_points,replace=False)
			pointcloud1 = self.data[item][choose_idx,:]
			pointcloud2 = self.data[item][choose_idx,:]    
		else:
			choose_idx1 = np.random.choice(range(self.num_points_CAD),self.num_points,replace=False)
			pointcloud1 = self.data[item][choose_idx1,:]
			choose_idx2 = np.random.choice(range(self.num_points_CAD),self.num_points,replace=False)
			pointcloud2 = self.data[item][choose_idx2,:]    


			
		if self.partial < 1:
			rand_xyz = uniform_2_sphere()
			pc1_centered = pointcloud1 - np.mean(pointcloud1, axis=0)
			dist_from_plane = np.dot(pc1_centered, rand_xyz)

			partial_idx = np.argsort(dist_from_plane)[0:int(self.num_points * self.partial)]
			pointcloud1 = pointcloud1[partial_idx,:]
		

		
		if self.gaussian_noise:
			pointcloud1 = jitter_pointcloud(pointcloud1)
			pointcloud2 = jitter_pointcloud(pointcloud2)
		
		# permutation
		pointcloud2 = np.random.permutation(pointcloud2)
		
		corr_mat_ab=None
		if (self.method == 'dcp_modified') or (self.method == 'pcrnet_corr'): # compute corr_mat if dcp_modified or pcrnet_corr
			corr_mat_ab = self.compute_corr_mat(pointcloud1,pointcloud2)
		
		pointcloud2 = rotation_ab.apply(pointcloud2) + np.expand_dims(translation_ab, axis=0)

		
		
		# return method-wise data:
		inputs = [item, pointcloud1, pointcloud2, R_ab, translation_ab, R_ba, translation_ba] # pc shape: (N,3)        
		
		if self.method == 'pointnetlk':
			return self.to_pointnetlk(*inputs)
		elif self.method == 'pcrnet':
			return self.to_pcrnet(*inputs)
		elif self.method == 'pcrnet_corr':
			return self.to_pcrnet_corr(*inputs, corr_mat_ab, euler_ab)
		elif self.method == 'rpmnet':
			return self.to_rpmnet(*inputs)
		elif (self.method == 'dcp') or (self.method == 'prnet'):
			pointcloud1 = pointcloud1.T  # changed to (3,N)
			pointcloud2 = pointcloud2.T  # changed to (3,N)

			return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
				   translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
				   euler_ab.astype('float32'), euler_ba.astype('float32')
		
		else: # dcp_modified
			pointcloud1 = pointcloud1.T  # changed to (3,N)
			pointcloud2 = pointcloud2.T  # changed to (3,N)

			return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
				   translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
				   euler_ab.astype('float32'), euler_ba.astype('float32'), corr_mat_ab.astype('int64')  
			

	def __len__(self):
		return self.data.shape[0]
	
	def compute_corr_mat(self, pointcloud1, pointcloud2):
 
		self.num_src = int(self.num_points * self.partial)
		self.num_tgt = self.num_points

		tree2=KDTree(pointcloud2)
		distance, col_idx_KDTree = tree2.query(pointcloud1) 
#         outlier_idx_KDTree = np.sort(np.where(distance > self.outl_thresh)[0] )

		corr_mat_ab_KDTree = np.zeros((self.num_tgt,self.num_src),dtype=np.int64)
		corr_mat_ab_KDTree[col_idx_KDTree,np.arange(self.num_src)]=1
		

#         helper.display_two_clouds_corr_mat(pointcloud1+1,pointcloud2,corr_mat_ab_KDTree,disp_time=60)


#         corr_mat_ab_KDTree[:,outlier_idx_KDTree] = 0
#         corr_mat_ab_KDTree[-1,outlier_idx_KDTree] = 1

		return corr_mat_ab_KDTree



	def to_rpmnet(self, *inputs):
		'''
		item           :- input to __getitem__() function
		pointcloud1    :- src    shape: (M,3)    M = N*partial
		pointcloud2    :- tgt    shape: (N,3)
		R_ab           :- R_ab * pointcloud1 + translation_ab = pointcloud2
		translation_ab :- same
		R_ba           :-  R_ab.T
		translation_ba :- -R_ba.dot(translation_ab)
		'''
		
		item, pointcloud1, pointcloud2, R_ab, translation_ab, R_ba, translation_ba  = inputs

		sample = {'points_raw': self.data[item, :, :], 'label': self.label[item], 'idx': np.array(item, dtype=np.int32)}

		sample['points_src'] = pointcloud1
		sample['points_ref'] = pointcloud2

		transform = np.concatenate((R_ab, translation_ab[:, None]), axis=1).astype(np.float32)

		sample['transform_gt'] = transform

		return sample

	def to_pointnetlk(self, *inputs):
		'''
		item           :- input to __getitem__() function
		pointcloud1    :- src    shape: (M,3)    M = N*partial
		pointcloud2    :- tgt    shape: (N,3)
		R_ab           :- R_ab * pointcloud1 + translation_ab = pointcloud2
		translation_ab :- same
		R_ba           :-  R_ab.T
		translation_ba :- -R_ba.dot(translation_ab)
		'''

		item, pointcloud1, pointcloud2, R_ab, translation_ab, R_ba, translation_ba = inputs

		train_classifier=False
		if train_classifier:
			return pointcloud1, self.label[item]
		else: 
			# Train PointNetLK

			transform_gt_4x4 = np.concatenate([np.concatenate((R_ab, translation_ab[:, None]), axis=1),
											   np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)], axis=0)
			return pointcloud1, pointcloud2, transform_gt_4x4

	def to_pcrnet(self, *inputs):
		'''
		item           :- input to __getitem__() function
		pointcloud1    :- src    shape: (M,3)    M = N*partial
		pointcloud2    :- tgt    shape: (N,3)
		R_ab           :- R_ab * pointcloud1 + translation_ab = pointcloud2
		translation_ab :- same
		R_ba           :-  R_ab.T
		translation_ba :- -R_ba.dot(translation_ab)
		'''
		
		item, pointcloud1, pointcloud2, R_ab, translation_ab, R_ba, translation_ba = inputs

		transform = np.concatenate((R_ab, translation_ab[:, None]), axis=1).astype(np.float32)

		return pointcloud1, pointcloud2

	def to_pcrnet_corr(self, *inputs):
		'''
		item           :- input to __getitem__() function
		pointcloud1    :- src    shape: (M,3)    M = N*partial
		pointcloud2    :- tgt    shape: (N,3)
		R_ab           :- R_ab * pointcloud1 + translation_ab = pointcloud2
		translation_ab :- same
		R_ba           :-  R_ab.T
		translation_ba :- -R_ba.dot(translation_ab)
		corr_mat       :- shape=(num_tgt,num_src)
		'''
		
		item, pointcloud1, pointcloud2, R_ab, translation_ab, R_ba, translation_ba, corr_mat, euler_ab = inputs

		transform = np.concatenate((R_ab, translation_ab[:, None]), axis=1).astype(np.float32)

		return pointcloud1, pointcloud2, corr_mat, R_ab.astype('float32'), translation_ab.astype('float32'), euler_ab.astype('float32')  # corr_mat shape: (num_tgt,num_src)


		
		


if __name__ == '__main__':
	n = 1024
	# n = 512
	# n = 100
	train = ModelNet40(n,partial=0.7,factor=1)
	# test = ModelNet40(n, 'test')



	for data in train:
		print(len(data))

		break


	load_data("train")
