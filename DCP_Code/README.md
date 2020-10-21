# PCR-CMU: Source Code for DCP Experiments

## Network Architecture for DCP Experiments:
<p align="center">
	<img src="https://github.com/tzodge/PCR-CMU/blob/main/DCP_Code/images/DCP_arch_v2.png" height="400">
</p>

## Before Training

`pip3 install -r requirements.txt`

## Training

Arguments:

`--exp_name`: Name of the experiment

`--num_points`: Number of points in the source and target point clouds (except for the partial case)

`--factor`: initial misalignment will be `pi / factor`

`--loss`: loss function to use. Use `cross_entropy_corr` to train with correspondence loss and `mse_transf` to train with transformation loss

`--partial`: num_points*partial will be removed from the source

`--cut_plane`: if True, the sampled points in the partial point cloud lie on one side of a random plane. If False, they are sampled randomly

### Exp 1.1
`python main.py --exp_name=exp1_1 --num_points=512 --factor=1 --loss=cross_entropy_corr`

### Exp 1.2

`python main.py --exp_name=exp1_2 --num_points=512 --factor=1 --loss=cross_entropy_corr --partial=0.3 --cut_plane`

### Exp 1.3

`python main.py --exp_name=exp1_3 --num_points=512 --factor=4 --loss=cross_entropy_corr`

To train models with transformation loss, use `--loss=mse_transf` instead.

## Evaluation

`--eval`: Add this argument to the above commands to evaluate a model.

`--model_path`: Path to the pretrained model to be evaluated.
