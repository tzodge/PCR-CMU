import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from util import npmat2euler

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
from model import PointNet
from model import iPCRNet
from losses import ChamferDistanceLoss
from losses import FrobeniusNormLoss
from data import ModelNet40

def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
	os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
	os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')

def create_transformation(rotation, translation):
	# rotation: 	[Bx3x3]
	# translation:  [Bx3x1]
	
	B = rotation.shape[0]
	last_row = torch.zeros(B, 1, 4)
	last_row[:, :, 3] = 1
	T = torch.cat([rotation, translation], axis=2)
	T = torch.cat([T, last_row], axis=1)
	return T.to(rotation.device)

class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()

def test_one_epoch(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	rotations_ab = []
	translations_ab = []
	rotations_ab_pred = []
	translations_ab_pred = []
	eulers_ab = []

	for i, data in enumerate(tqdm(test_loader)):
		# if i>5: break
		template, source, rotation_ab, translation_ab, euler_ab = data
		# source = rotation_ab * template + translation_ab

		template = template.to(device).float()
		source = source.to(device).float()
		rotation_ab = rotation_ab.to(device).float()
		translation_ab = translation_ab.to(device).float()
		# eulers_ab = eulers_ab.float()
		translation_ab = translation_ab - torch.mean(source, dim=1)

		# mean substraction
		source = source - torch.mean(source, dim=1, keepdim=True)
		template = template - torch.mean(template, dim=1, keepdim=True)

		output = model(template, source)

		# source = rotation_ab_pred*template + translation_ab_pred
		rotation_ba_pred = output['est_R']
		translation_ba_pred = output['est_t']
		rotation_ab_pred = output['est_R'].permute(0, 2, 1)
		translation_ab_pred = -torch.bmm(output['est_R'].permute(0, 2, 1), output['est_t'].permute(0, 2, 1)).permute(0, 2, 1)		# -R^T * t

		rotations_ab.append(rotation_ab.detach().cpu().numpy())
		translations_ab.append(translation_ab.detach().cpu().numpy())
		rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
		translations_ab_pred.append(translation_ab_pred.view(-1, 3).detach().cpu().numpy())
		eulers_ab.append(euler_ab.numpy())

		try:
			loss_val = ChamferDistanceLoss()(template, output['transformed_source'])
		except:
			igt = create_transformation(rotation_ab, translation_ab)
			predicted = create_transformation(rotation_ba_pred, translation_ba_pred)
			loss_val = FrobeniusNormLoss()(predicted, igt)

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count

	rotations_ab = np.concatenate(rotations_ab, axis=0)
	translations_ab = np.concatenate(translations_ab, axis=0)
	rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
	translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
	eulers_ab = np.concatenate(eulers_ab, axis=0)

	return test_loss, rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, eulers_ab

def test(args, net, test_loader, textio):
	test_loss, test_rotations_ab, test_translations_ab, \
	test_rotations_ab_pred, test_translations_ab_pred, \
	test_eulers_ab = test_one_epoch(args, net, test_loader)
	

	test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
	test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
	test_r_rmse_ab = np.sqrt(test_r_mse_ab)
	test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
	test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
	test_t_rmse_ab = np.sqrt(test_t_mse_ab)
	test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

	textio.cprint('==FINAL TEST==')
	textio.cprint('A--------->B')
	textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
				  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f, Corr_Accuracy: %f'
				  % (-1, test_loss, test_r_mse_ab, test_r_rmse_ab,
					 test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab, test_corr_accuracy))

def train_one_epoch(device, model, train_loader, optimizer):
	model.train()
	train_loss = 0.0
	pred  = 0.0
	count = 0
	rotations_ab = []
	translations_ab = []
	rotations_ab_pred = []
	translations_ab_pred = []
	eulers_ab = []

	for i, data in enumerate(tqdm(train_loader)):
		# if i > 5: break
		template, source, rotation_ab, translation_ab, euler_ab = data
		# source = rotation_ab * template + translation_ab

		template = template.to(device).float()
		source = source.to(device).float()
		rotation_ab = rotation_ab.to(device).float()
		translation_ab = translation_ab.to(device).float()
		# eulers_ab = eulers_ab.float()
		translation_ab = translation_ab - torch.mean(source, dim=1)

		# mean substraction
		source = source - torch.mean(source, dim=1, keepdim=True)
		template = template - torch.mean(template, dim=1, keepdim=True)

		output = model(template, source)
		
		# source = rotation_ab_pred*template + translation_ab_pred
		rotation_ab_pred = output['est_R'].permute(0, 2, 1)
		translation_ab_pred = -torch.bmm(output['est_R'].permute(0, 2, 1), output['est_t'].permute(0, 2, 1)).permute(0, 2, 1)		# -R^T * t
		
		rotations_ab.append(rotation_ab.detach().cpu().numpy())
		translations_ab.append(translation_ab.detach().cpu().numpy())
		rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
		translations_ab_pred.append(translation_ab_pred.view(-1, 3).detach().cpu().numpy())
		eulers_ab.append(euler_ab.numpy())

		loss_val = ChamferDistanceLoss()(template, output['transformed_source'])

		# forward + backward + optimize
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		train_loss += loss_val.item()
		count += 1

	train_loss = float(train_loss)/count
	rotations_ab = np.concatenate(rotations_ab, axis=0)
	translations_ab = np.concatenate(translations_ab, axis=0)
	rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
	translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
	eulers_ab = np.concatenate(eulers_ab, axis=0)

	return train_loss, rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, eulers_ab

def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
	learnable_params = filter(lambda p: p.requires_grad, model.parameters())
	if args.optimizer == 'Adam':
		optimizer = torch.optim.Adam(learnable_params)
	else:
		optimizer = torch.optim.SGD(learnable_params, lr=0.1)

	if checkpoint is not None:
		min_loss = checkpoint['min_loss']
		optimizer.load_state_dict(checkpoint['optimizer'])

	best_test_loss = np.inf
	best_test_r_mse_ab = np.inf
	best_test_r_rmse_ab = np.inf
	best_test_r_mae_ab = np.inf
	best_test_t_mse_ab = np.inf
	best_test_t_rmse_ab = np.inf
	best_test_t_mae_ab = np.inf

	for epoch in range(args.start_epoch, args.epochs):
		train_loss, train_rotations_ab, train_translations_ab, train_rotations_ab_pred, train_translations_ab_pred, train_eulers_ab = train_one_epoch(args.device, model, train_loader, optimizer)
		test_loss, test_rotations_ab, test_translations_ab, test_rotations_ab_pred, test_translations_ab_pred, test_eulers_ab = test_one_epoch(args.device, model, test_loader)

		train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
		train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
		train_r_rmse_ab = np.sqrt(train_r_mse_ab)
		train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
		train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
		train_t_rmse_ab = np.sqrt(train_t_mse_ab)
		train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

		test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
		test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
		test_r_rmse_ab = np.sqrt(test_r_mse_ab)
		test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
		test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
		test_t_rmse_ab = np.sqrt(test_t_mse_ab)
		test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

		if test_loss<best_test_loss:
			best_test_loss = test_loss
			best_test_r_mse_ab = test_r_mse_ab
			best_test_r_rmse_ab = test_r_rmse_ab
			best_test_r_mae_ab = test_r_mae_ab

			best_test_t_mse_ab = test_t_mse_ab
			best_test_t_rmse_ab = test_t_rmse_ab
			best_test_t_mae_ab = test_t_mae_ab

			snap = {'epoch': epoch + 1,
					'model': model.state_dict(),
					'min_loss': best_test_loss,
					'optimizer' : optimizer.state_dict(),}
			torch.save(snap, 'checkpoints/%s/models/best_model_snap.t7' % (args.exp_name))
			torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.t7' % (args.exp_name))
			torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/best_ptnet_model.t7' % (args.exp_name))

		torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % (args.exp_name))
		torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % (args.exp_name))
		torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/ptnet_model.t7' % (args.exp_name))
		
		boardio.add_scalar('Train Loss', train_loss, epoch)
		boardio.add_scalar('Test Loss', test_loss, epoch)
		boardio.add_scalar('Best Test Loss', best_test_loss, epoch)

		boardio.add_scalar('A->B/train/rotation/MSE', train_r_mse_ab, epoch)
		boardio.add_scalar('A->B/train/rotation/RMSE', train_r_rmse_ab, epoch)
		boardio.add_scalar('A->B/train/rotation/MAE', train_r_mae_ab, epoch)
		boardio.add_scalar('A->B/train/translation/MSE', train_t_mse_ab, epoch)
		boardio.add_scalar('A->B/train/translation/RMSE', train_t_rmse_ab, epoch)
		boardio.add_scalar('A->B/train/translation/MAE', train_t_mae_ab, epoch)
		
		boardio.add_scalar('A->B/test/rotation/MSE', test_r_mse_ab, epoch)
		boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
		boardio.add_scalar('A->B/test/rotation/MAE', test_r_mae_ab, epoch)
		boardio.add_scalar('A->B/test/translation/MSE', test_t_mse_ab, epoch)
		boardio.add_scalar('A->B/test/translation/RMSE', test_t_rmse_ab, epoch)
		boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)

		boardio.add_scalar('A->B/best_test/rotation/MSE', best_test_r_mse_ab, epoch)
		boardio.add_scalar('A->B/best_test/rotation/RMSE', best_test_r_rmse_ab, epoch)
		boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
		boardio.add_scalar('A->B/best_test/translation/MSE', best_test_t_mse_ab, epoch)
		boardio.add_scalar('A->B/best_test/translation/RMSE', best_test_t_rmse_ab, epoch)
		boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)

		textio.cprint('==TRAIN==')
		textio.cprint('A--------->B')
		textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
					  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
					  % (epoch, train_loss, train_r_mse_ab,
						 train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))

		textio.cprint('==TEST==')
		textio.cprint('A--------->B')
		textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
					  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
					  % (epoch, test_loss, test_r_mse_ab,
						 test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))

		textio.cprint('==BEST TEST==')
		textio.cprint('A--------->B')
		textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
					  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
					  % (epoch, best_test_loss, best_test_r_mse_ab, best_test_r_rmse_ab,
						 best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab))


def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--exp_name', type=str, default='exp_ipcrnet_pts100_factor1', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=100, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')
	parser.add_argument('--gaussian_noise', default=False, type=bool)
	parser.add_argument('--unseen', default=False, type=bool)
	parser.add_argument('--factor', default=4, type=int)

	# settings for PointNet
	parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
						help='train pointnet (default: tune)')
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')

	# settings for on training
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=32, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--epochs', default=200, type=int,
						metavar='N', help='number of total epochs to run')
	parser.add_argument('--start_epoch', default=0, type=int,
						metavar='N', help='manual epoch number (useful on restarts)')
	parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
						metavar='METHOD', help='name of an optimizer (default: Adam)')
	parser.add_argument('--resume', default='', type=str,
						metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
	parser.add_argument('--pretrained', default='', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
	args = options()

	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
	_init_(args)

	textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
	textio.cprint(str(args))

	
	train_loader = DataLoader(
		ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
				   unseen=args.unseen, factor=args.factor, method='pcrnet'),
		batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6)
	test_loader = DataLoader(
		ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
				   unseen=args.unseen, factor=args.factor, method='pcrnet'),
		batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=6)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNet Model.
	ptnet = PointNet(emb_dims=args.emb_dims)
	model = iPCRNet(feature_model=ptnet)
	model = model.to(args.device)

	checkpoint = None
	if args.resume:
		assert os.path.isfile(args.resume)
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model.to(args.device)

	if args.eval:
		test(args, model, test_loader, textio)
	else:
		train(args, model, train_loader, test_loader, boardio, textio, checkpoint)

if __name__ == '__main__':
	main()