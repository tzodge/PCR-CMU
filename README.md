# Correspondence Matrices are Underrated  
<!-- Recent work in deep learning has made point cloud registration faster as compared to existing methods. Out of the two interdependent parameters, correspondence and transformation we observed that correspondence is more robust parameter for registration. Even then many of the existing learning based registration methods like PCRNet, RPMNet, DCP train the network to learn the transformation between the input point clouds. In the work [Correspondence Matrices are Underrated](), we empirically show that if these networks are trained to explicitly learn correspondence instead of transformations can register more accurately, can deal with partial point clouds and can deal with larger misalignments between the input point clouds.

| ![Image](/images/corr_vs_transf.png) | 
|:--:| 
| Correspondence Vs Transformation | -->

This work is based on our observation that correspondence is more robust parameter for point cloud registration than transformations. 

We show that an existing deep learning based method which trains the network to learn transformations can converge faster, can register more accurately, and can register partial point clouds if trained to learn correspondence.

This is demonstrated by comparing methods (Trained to learn transformation) like DCP, PCRNet, and RPMNet with method_corr (Trained to learn correspondence) DCP_corr, PCRNet_corr, and RPMNet_corr respectively.

| DCP Vs DCP_corr | 
|:--:| 
| ![Image](/images/DCP_charts.png) | 

| PCRNet Vs PCRNet_corr | 
|:--:| 
| ![Image](/images/PCRNet_charts.png) | 

| RPMNet Vs RPMNet_corr | 
|:--:| 
| ![Image](/images/RPMNet_charts.png) | 

