# PCR-CMU Source Code for RPMNet Experiments

## Network Architecture for PCRNet Experiments:
<p align="center">
	<img src="https://github.com/tzodge/PCR-CMU/blob/main/RPMNet_Code/images/RPMNet_arch_v3.png" height="300">
</p>


## Contents
pretrained: Pretrained models for all the RPMNet experiments mentioned in the paper.

src: Source Code

requirements.txt: Libraries Required

## Before Training

`pip3 install -r requirements.txt`

`cd src/`

## Training Commands
### exp 2.1
`python train.py --loss_method corrloss --noise_type clean --rot_mag 180 --train_batch_size 4 --lr 1e-3 --val_batch_size 4 --gpu 0 --name corrloss_clean_180deg`

### exp 2.2
`python train.py --loss_method corrloss --noise_type partial_identical --partial 0.7 --rot_mag 180 --train_batch_size 4 --lr 1e-3 --val_batch_size 4 --gpu 0 --name corrloss_partial_180deg`

### exp 2.3
`python train.py --loss_method corrloss --noise_type partial_identical --partial 0.7 --rot_mag 45 --train_batch_size 4 --lr 1e-3 --val_batch_size 4 --gpu 0 --name corrloss_partial_45deg`

### NOTE: 
Use `--loss_method=rpmloss` to train with transformation loss as mentioned in the RPMNet paper.

## Evaluation Commands
### exp 2.1
`python eval.py --noise_type clean --rot_mag 180 --gpu 0 --val_batch_size 4 --resume ../pretrained/full_180_corrloss_best.pth`

### exp 2.2
`python eval.py --noise_type partial_identical --partial 0.7 --rot_mag 180 --gpu 0 --val_batch_size 4 --resume ../pretrained/partial_180_corrloss_best.pth`

### exp 2.3
`python eval.py --noise_type partial_identical --partial 0.7 --rot_mag 45 --gpu 0 --val_batch_size 4 --resume ../pretrained/partial_45_corrloss_best.pth`
