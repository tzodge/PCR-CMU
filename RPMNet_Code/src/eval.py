"""Evaluate RPMNet.

Example Usages:
    1. Evaluate RPMNet
        python eval.py --noise_type clean --resume [path-to-model.pth]
"""
from collections import defaultdict
import json
import os
import pickle
import time
from typing import Dict, List

import numpy as np
import open3d  # Need to import before torch
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import torch

from arguments import rpmnet_eval_arguments
from common.misc import prepare_logger
from common.torch import dict_all_to_device, CheckPointManager, to_numpy
from common.math import se3
from common.math_torch import se3
from common.math.so3 import dcm2euler
from data_loader.datasets import get_test_datasets
import models.rpmnet


def compute_metrics(data: Dict, pred_transforms, perm_matrices=None) -> Dict:
    """Compute metrics required in the paper
    """

    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)
    
    

    with torch.no_grad():
        pred_transforms = pred_transforms
        gt_transforms = data['transform_gt']
        points_src = data['points_src'][..., :3]
        points_ref = data['points_ref'][..., :3]
        points_raw = data['points_raw'][..., :3]

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        # Modified Chamfer distance
        src_transformed = se3.transform(pred_transforms, points_src)
        ref_clean = points_raw
        src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
        dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)


        # computing percentage of correct correspondences        
        if perm_matrices is not None:

            scores_pred = perm_matrices #b,m,n
            scores_gt    = data['corr_mat'] # b,m,n

            corr_mat_pred = scores_pred.detach().cpu().numpy()     # b,m,n    
            col_idx_pred = np.argmax(corr_mat_pred,axis=-1) 
            corr_mat_gt = scores_gt.detach().cpu().numpy()     # b,m,n   
            col_idx_gt = np.argmax(corr_mat_gt,axis=-1)        # b,m

            correct_mask = (col_idx_gt == col_idx_pred)*1      # b,m
            correct_corr = np.mean(correct_mask,axis=1) # b
       
        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_numpy(t_mse),
            't_mae': to_numpy(t_mae),
            'err_r_deg': to_numpy(residual_rotdeg),
            'err_t': to_numpy(residual_transmag),
            'chamfer_dist': to_numpy(chamfer_dist),
            'correct_corr': correct_corr
        }

    return metrics


def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def print_metrics(logger, summary_metrics: Dict, losses_by_iteration: List = None,
                  title: str = 'Metrics'):
    """Prints out formated metrics to logger"""

    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])
        logger.info('Losses by iteration: {}'.format(losses_all_str))

    logger.info('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    logger.info('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    logger.info('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))
    logger.info('Chamfer error: {:.7f}(mean-sq)'.format(
        summary_metrics['chamfer_dist']))
    if 'correct_corr' in summary_metrics.keys():
        logger.info('Correct inlier correspondences: {:.7f}(percent)'.format(
            summary_metrics['correct_corr']*100))


def evaluate(data_loader, model: torch.nn.Module):
    """ Evaluates the model's prediction against the groundtruth """
    _logger.info('Starting evaluation...')
    with torch.no_grad():
        all_test_metrics_np = defaultdict(list)
        for test_data in data_loader:
            dict_all_to_device(test_data, _device)
            pred_test_transforms, endpoints = model(test_data, _args.num_reg_iter)
            test_metrics = compute_metrics(test_data, pred_test_transforms[-1],endpoints['perm_matrices'][-1])
            for k in test_metrics:
                all_test_metrics_np[k].append(test_metrics[k])
        all_test_metrics_np = {k: np.concatenate(all_test_metrics_np[k]) for k in all_test_metrics_np}

    summary_metrics = summarize_metrics(all_test_metrics_np)
    print_metrics(_logger, summary_metrics, title='Evaluation results')


def get_model():
    _logger.info('Computing transforms using {}'.format(_args.method))
    if _args.method == 'rpmnet':
        assert _args.resume is not None
        model = models.rpmnet.get_model(_args)
        model.to(_device)
        saver = CheckPointManager(os.path.join(_log_path, 'ckpt', 'models'))
        saver.load(_args.resume, model)
    else:
        raise NotImplementedError
    return model


def main():
    # Load data_loader
    test_dataset = get_test_datasets(_args)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=_args.val_batch_size, shuffle=False)

    model = get_model()
    model.eval()
    evaluate(test_loader, model)
    _logger.info('Finished')


if __name__ == '__main__':
    # Arguments and logging
    parser = rpmnet_eval_arguments()
    _args = parser.parse_args()
    _logger, _log_path = prepare_logger(_args, log_path=_args.eval_save_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    if _args.gpu >= 0 and (_args.method == 'rpm' or _args.method == 'rpmnet'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
        _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        _device = torch.device('cpu')

    main()
