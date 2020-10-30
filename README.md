# Correspondence Matrices are Underrated  

[**Paper**](http://biorobotics.ri.cmu.edu/papers/paperUploads/PID6659733.pdf) | [**Supplementary**]() | [**Project Website**]() | [**Video**](https://youtu.be/wCy_GDfX2CA)

This is the source code repository of the paper "Correspondence Matrices are Underrated" accepted at the 8th International Conference on 3D Vision, 2020.

Source Code Authors: [Tejas Zodage](https://github.com/tzodge), [Rahul Chakwate](https://github.com/ruc98), [Vinit Sarode](https://github.com/vinits5)

<!-- Recent work in deep learning has made point cloud registration faster as compared to existing methods. Out of the two interdependent parameters, correspondence and transformation we observed that correspondence is more robust parameter for registration. Even then many of the existing learning based registration methods like PCRNet, RPMNet, DCP train the network to learn the transformation between the input point clouds. In the work [Correspondence Matrices are Underrated](), we empirically show that if these networks are trained to explicitly learn correspondence instead of transformations can register more accurately, can deal with partial point clouds and can deal with larger misalignments between the input point clouds.

| ![Image](/images/corr_vs_transf.png) | 
|:--:| 
| Correspondence Vs Transformation | -->


### Methodology:

<p align="center">
	<img src="https://github.com/tzodge/PCR-CMU/blob/main/images/framework.gif" height="600">
</p>


This work is based on our observation that correspondence is more robust parameter for point cloud registration than transformations. 
We show that an existing deep learning based method which trains the network to learn transformations can converge faster, can register more accurately, and can register partial point clouds if trained to learn correspondence.
This is demonstrated by comparing methods (Trained to learn transformation) like DCP, PCRNet, and RPMNet with method_corr (Trained to learn correspondence) DCP_corr, PCRNet_corr, and RPMNet_corr respectively.
<!--
### Results:

<p align="center">
	<img src="https://github.com/tzodge/PCR-CMU/blob/main/images/results1.png" height="300">
</p>

<p align="center">
	<img src="https://github.com/tzodge/PCR-CMU/blob/main/images/results2.png" height="300">
</p>
 -->
### Citation:
```
@InProceedings{tejas2020cmu,
    author       = "Tejas Zodage, Rahul Chakwate, Vinit Sarode, Rangaprasad Arun Srivatsan and Howie Choset",
    title        = "Correspondence Matrices are Underrated",
    booktitle    = "International Conference on 3D Vision (3DV)",
    month        = "Nov.",
    year         = "2020",
  }
```

### Usage:

We demonstrate our approach on three recent network architectures: DCP, PCRNet and RPMNet.

Please refer to the ReadMe sections of the corresponding folders: [**DCP_Code**](https://github.com/tzodge/PCR-CMU/tree/main/DCP_Code), [**PCRNet_Code**](https://github.com/tzodge/PCR-CMU/tree/main/PCRNet_Code) and [**RPMNet_Code**](https://github.com/tzodge/PCR-CMU/tree/main/RPMNet_Code).

### License:

This project is release under the MIT License.

### Pretrained Models:

The pretrained models for all three networks can be downloaded from [here](https://drive.google.com/drive/folders/1PwFLCNHiL66jL3KySa8msJ_btIvevqW4?usp=sharing).

### Acknowledgement:

We would like to thank the authors of [DCP](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf), [PCRNet](https://arxiv.org/abs/1908.07906), [RPM-Net](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yew_RPM-Net_Robust_Point_Matching_Using_Learned_Features_CVPR_2020_paper.pdf), and [PointNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) for making their codes available.


### Graphical Results:

| DCP Vs DCP_corr | 
|:--:| 
| ![Image](/images/DCP_charts.png) | 

| PCRNet Vs PCRNet_corr | 
|:--:| 
| ![Image](/images/PCRNet_charts.png) | 

| RPMNet Vs RPMNet_corr | 
|:--:| 
| ![Image](/images/RPMNet_charts.png) | 

