#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from scipy.spatial import KDTree
import helper 
import matplotlib.pyplot as plt
from helper import print_rounded
import transforms3d.axangles as t3d_axang
import MinkowskiEngine as ME
import open3d as o3d

def make_open3d_point_cloud(xyz, color=None):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	if color is not None:
		pcd.colors = o3d.utility.Vector3dVector(color)
	return pcd
# The code is referred from: https://github.com/WangYueFt/dcp 

def download():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))


def load_data(partition):
	download()
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, 'data')
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


def translate_pointcloud(pointcloud):
	xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
	xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

	translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
	return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.02, clip=0.05):
	N, C = pointcloud.shape
	pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
	return pointcloud

def register_clouds(pointcloud_a,pointcloud_b,corr_mat_ab):
	'''
	input:
		pointcloud_a = mx3  --> source
		pointcloud_b = nx3  --> target
		corr_mat_ab = nxm    
	output:
		R = 3x3
		t = 3x1
		such that 
		e = ||R*pointcloud_a.T + t - pointcloud_b.T . corr_mat_ab|| ^2 is minimum    
	''' 
	pointcloud_a_corr = (pointcloud_b.T.dot(corr_mat_ab)).T
	pointcloud_a_corr_cent = pointcloud_a_corr - pointcloud_a_corr.mean(axis=0)

	pointcloud_a_cent = pointcloud_a - pointcloud_a.mean(axis=0)
	
	H = pointcloud_a_cent.T.dot(pointcloud_a_corr_cent)
	U, S, Vt = np.linalg.svd(H)
	R =  Vt.T.dot(U.T)

	if np.linalg.det(R) < 0:
			Vt[2,:] *= -1
			R = np.dot(Vt.T, U.T)
		# t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
	t = pointcloud_a_corr.mean(axis=0).reshape(-1,1) - R.dot(pointcloud_a.mean(axis=0).reshape(-1,1)) 
	return R,t


class ModelNet40(Dataset):
	# partial = 0.1 ==> num_points*partial will be removed.

	def __init__(self, num_points, partition='train',\
				 gaussian_noise=False, unseen=False, \
				 factor=4, same_pointclouds=True,\
				 partial=0.0,\
				 debug=False,
				 cut_plane=False,
				 outliers=0.0,
				 single_category=False): 

		self.data, self.label = load_data(partition)
		print (len(self.data),"len(self.data)")
		self.partition = partition
		self.gaussian_noise = gaussian_noise
		self.same_pointclouds = same_pointclouds
		self.unseen = unseen
		self.label = self.label.squeeze()
		self.factor = factor
		self.partial = partial
		self.num_tgt = num_points
		self.num_src = int(num_points*(1-partial))
		self.debug = debug
		self.cut_plane = cut_plane
		self.outliers = outliers
		self.outl_thresh = 1
		self.num_points_CAD = len(self.data[0])
		self.single_category = single_category

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
		seed_val = torch.rand(1,)[0]*1e7
		np.random.seed(int(seed_val ))

 
		choose_idx = np.random.choice(range(self.num_points_CAD),self.num_tgt,replace=False)
		pointcloud1 = self.data[item][choose_idx,:]
		choose_idx = np.random.choice(range(self.num_points_CAD),self.num_tgt,replace=False)
		pointcloud2 = self.data[item][choose_idx,:] 

		if self.gaussian_noise:
			pointcloud1 = jitter_pointcloud(pointcloud1)
			pointcloud2 = jitter_pointcloud(pointcloud2)
		if self.partition != 'train':
			np.random.seed(item)
 
		axis = np.random.rand(3,) - np.array([0.5,0.5,0.5])  
		axis = axis/np.linalg.norm(axis)

		# using axis angles instead of euler angles to avoid gimbal lock. Slight variation from DCP dataloader.
		# angle from [-pi/factor to +pi/factor]
		angle = 2*(np.random.uniform()-0.5) * np.pi / self.factor

		Rot_instance = Rotation.from_rotvec(axis*angle) 
		R_ab = Rot_instance.as_dcm()

		[anglez,angley,anglex] = Rot_instance.as_euler('zyx')             
 
		R_ba = R_ab.T
		translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
								   np.random.uniform(-0.5, 0.5)]) 
		translation_ba = -R_ba.dot(translation_ab)

		pointcloud1 = pointcloud1.T
		pointcloud2 = pointcloud2.T

		rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
		if self.same_pointclouds:
			pointcloud2 = np.copy(pointcloud1)
			if self.gaussian_noise:
				pointcloud2 = jitter_pointcloud(pointcloud2)


		euler_ab = np.asarray([anglez, angley, anglex])
		euler_ba = -euler_ab[::-1]

		col_idx =  np.arange(self.num_tgt)
		col_idx=np.random.permutation(col_idx)    
		pointcloud2_shuff = pointcloud2[:,col_idx] 
		


		if self.partial > 0.001: 
			num_points_delete = self.num_tgt - self.num_src    
			if self.cut_plane:
				plane_coeff = np.random.rand(4,)-0.5
				plane_coeff[3]=0
				above_points,above_idx = helper.points_above_plane(pointcloud1.T,plane_coeff)
				delete_idx = above_idx[0:num_points_delete]

			else:                
				delete_idx = np.random.choice(range(self.num_tgt),num_points_delete,replace=False)
			
			pointcloud1 = np.delete(pointcloud1,delete_idx,axis=1)


		# generating ground-truth correspondence matrix
		tree2=KDTree(pointcloud2_shuff.T)
		distance, col_idx_KDTree = tree2.query(pointcloud1.T) 
		outlier_idx_KDTree = np.sort(np.where(distance > self.outl_thresh)[0] )

 
		corr_mat_ab_KDTree = np.zeros((self.num_tgt ,self.num_src),dtype=np.int64)
		corr_mat_ab_KDTree[col_idx_KDTree,np.arange(self.num_src)]=1
	   
		corr_mat_ab_KDTree[:,outlier_idx_KDTree] = 0
		corr_mat_ab_KDTree[-1,outlier_idx_KDTree] = 1

		corr_mat_ab = corr_mat_ab_KDTree
		pointcloud2_shuff = rotation_ab.apply(pointcloud2_shuff.T).T + np.expand_dims(translation_ab, axis=1)



		R_svd,t_svd = register_clouds(pointcloud1.T,pointcloud2_shuff.T,corr_mat_ab)
		R_ab = R_svd
		translation_ab = t_svd.flatten()

		if self.debug:
		
			pointcloud2_transf = R_ba.dot(pointcloud2_shuff) + translation_ba.reshape(3,1)         
			pointcloud1_transf = R_ab.dot(pointcloud1) + translation_ab.reshape(3,1)         
 
			helper.display_two_clouds_corr_mat(pointcloud1_transf.T,pointcloud2_shuff.T,corr_mat_ab,disp_time=60)


		return pointcloud1.astype('float32'), pointcloud2_shuff.astype('float32'), R_ab.astype('float32'), \
			   translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
			   euler_ab.astype('float32'), euler_ba.astype('float32'), col_idx.astype('int64'), corr_mat_ab.astype('int64')  

	def __len__(self):
		return self.data.shape[0]



if __name__ == '__main__':
	n = 512
	train = ModelNet40(n,same_pointclouds=True,partial=0.3,cut_plane=True,debug=True,outliers=0.0,factor=1)
	# test = ModelNet40(n, 'test')

	for data in train:
		print(len(data))		
		break

	load_data("train")