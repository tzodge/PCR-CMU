# PCR-CMU: Source Code for PCRNet Experiments

## Network Architecture for PCRNet Experiments:
<p align="center">
	<img src="https://github.com/tzodge/PCR-CMU/blob/main/PCRNet_Code/images/PCRNet_arch_v2.png" height="300">
</p>

## Before training:
> pip3 install -r requirements.txt

## Training:
Arguments:
1. factor: Used to indicate initial misalignment range [-180/factor, 180/factor]
2. num_points: Number of points in template and source point clouds.
3. exp_name: Name of the log file created during training.


### Section 6.3, Experiment 3.1:
Points in Source: 1024, Points in Template: 1024 and Initial Misalignment Range: [-45, 45] degrees.

>python train_pcrnet_corr.py --factor 4 --num_points 1024 --exp_name exp_pcrnet_corr_1024_4

>python train_pcrnet.py --factor 4 --num_points 1024 --exp_name exp_pcrnet_vanilla_1024_4

### Section 6.3, Experiment 3.2:
Points in Source: 100, Points in Template: 100 and Initial Misalignment Range: [-45, 45] degrees.

>python train_pcrnet_corr.py --factor 4 --num_points 100 --exp_name exp_pcrnet_corr_100_4

>python train_pcrnet.py --factor 4 --num_points 100 --exp_name exp_pcrnet_vanilla_100_4

### Section 6.3, Experiment 3.3:
Points in Source: 100, Points in Template: 100 and Initial Misalignment Range: [-180, 180] degrees.

>python train_pcrnet_corr.py --factor 1 --num_points 100 --exp_name exp_pcrnet_corr_100_1

>python train_pcrnet.py --factor 1 --num_points 100 --exp_name exp_pcrnet_vanilla_100_1


## Evaluation:
Arguments:
1. eval: Set this argument as True to evaluate the pretrained model.
2. pretrained: Provide the complete path of the trained model.

### Section 6.3, Experiment 3.1:
> python train_pcrnet_corr.py --factor 4 --num_points 1024 --pretrained checkpoints/exp_pcrnet_corr_1024_4/models/model.best.t7 --exp_name exp_pcrnet_corr_1024_4_eval --eval True

>python train_pcrnet.py --factor 4 --num_points 1024 --pretrained checkpoints/exp_pcrnet_vanilla_1024_4/models/model.best.t7 --exp_name exp_pcrnet_vanilla_1024_4_eval --eval True

### Section 6.3, Experiment 3.2:
> python train_pcrnet_corr.py --factor 4 --num_points 100 --pretrained checkpoints/exp_pcrnet_corr_100_4/models/model.best.t7 --exp_name exp_pcrnet_corr_100_4_eval --eval True

>python train_pcrnet.py --factor 4 --num_points 100 --pretrained checkpoints/exp_pcrnet_vanilla_100_4/models/model.best.t7 --exp_name exp_pcrnet_vanilla_100_4_eval --eval True

### Section 6.3, Experiment 3.3:
> python train_pcrnet_corr.py --factor 1 --num_points 100 --pretrained checkpoints/exp_pcrnet_corr_100_1/models/model.best.t7 --exp_name exp_pcrnet_corr_100_1_eval --eval True

>python train_pcrnet.py --factor 1 --num_points 100 --pretrained checkpoints/exp_pcrnet_vanilla_100_1/models/model.best.t7 --exp_name exp_pcrnet_vanilla_100_1_eval --eval True

### Pretrained Models

The pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1PwFLCNHiL66jL3KySa8msJ_btIvevqW4?usp=sharing).

### Acknowledgement

We thank PCRNet authors for sharing their code. Their original code can be found at

PCRNet Repository: [https://github.com/vinits5/pcrnet_pytorch](https://github.com/vinits5/pcrnet_pytorch)
