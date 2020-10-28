# PCR-CMU Source Code for RPMNet Experiments

## Network Architecture for RPMNet Experiments:
<p align="center">
	<img src="https://github.com/tzodge/PCR-CMU/blob/main/RPMNet_Code/images/RPMNet_arch_v3.png" height="400">
</p>


## Before Training

`pip3 install -r requirements.txt`

`cd src/`

## Training

Arguments:

`--num_points`: Number of points to sample.

`--loss_method`: The loss type to train the model. Use `corrloss` for correspondence loss and use `rpmloss` for default loss function mentioned in RPMNet paper.

`--noise_type`: Type of noise & sampling in point clouds. `clean` refers to no noise & identical sampling. `partial_identical` refers to partial source & identical sampling.

`--partial`: `num_points*partial` points will be kept in the source. Only required if `--noise_type=partial_identical`

`--rot_mag`: Maximum initial misalignment (in deg)

`--name`: Name of the experiment

### Experiment 2.1
`python train.py --num_points 1024 --loss_method corrloss --noise_type clean --rot_mag 180 --name corrloss_clean_180deg`

### Experiment 2.2
`python train.py --num_points 1024 --loss_method corrloss --noise_type partial_identical --partial 0.7 --rot_mag 180 --name corrloss_partial_180deg`

### Experiment 2.3
`python train.py --num_points 1024 --loss_method corrloss --noise_type partial_identical --partial 0.7 --rot_mag 45 --name corrloss_partial_45deg`

### NOTE: 
Use `--loss_method=rpmloss` to train with transformation loss as mentioned in the RPMNet paper.

## Evaluation

`--resume`: Path to the pretrained model to evaluate

### Experiment 2.1
`python eval.py --noise_type clean --rot_mag 180 --resume ./<pretrained_path>/full_180_corrloss_best.pth`

### Experiment 2.2
`python eval.py --noise_type partial_identical --partial 0.7 --rot_mag 180 --resume ./<pretrained_path>/partial_180_corrloss_best.pth`

### Experiment 2.3
`python eval.py --noise_type partial_identical --partial 0.7 --rot_mag 45 --resume ./<pretrained_path>/partial_45_corrloss_best.pth`
