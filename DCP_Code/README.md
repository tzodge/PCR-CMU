# PCR-CMU: Source Code for DCP Experiments

## Network Architecture for DCP Experiments:
<p align="center">
	<img src="https://github.com/tzodge/PCR-CMU/blob/main/DCP_Code/images/DCP_arch_v2.png" height="400">
</p>

## Before Training

`pip3 install -r requirements.txt`

## Training

### Exp 1.1
`python main.py --exp_name=exp1_1 --num_points=512 --factor=1 --loss=cross_entropy_corr --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd  --batch_size=16`

### Exp 1.2

`python main.py --exp_name=exp1_2 --num_points=512 --factor=1 --partial=0.3 --cut_plane --loss=cross_entropy_corr --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd  --batch_size=16`

where 

`--partial`: num_points*partial will be removed.

`--cut_plane`: if True, the sampled points in the partial point cloud lie on one side of a random plane. If False, they are sampled randomly.

### Exp 1.3

`python main.py --exp_name=exp1_3 --num_points=512 --factor=4 --loss=cross_entropy_corr --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd  --batch_size=16`

To train models with transformation loss, use `--loss=mse_transf` instead.

## Evaluation

Add `--eval` argument to the above commands to evaluate a model.

Pass the appropriate `--model_path`.
