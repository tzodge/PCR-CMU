# Correspondence Matrices are Underrated  
Recent work in deep learning has made point cloud registration faster as compared to existing methods. Out of the two interdependent parameters, correspondence and transformation we observed that correspondence is more robust parameter for registration. Even then many of the existing learning based methods train the network to learn the transformation between the input point clouds. In the work [Correspondence Matrices are Underrated](), we empirically show that if these networks are trained to explicitly learn correspondence instead of transformations can register more accurately, can deal with partial point clouds and can deal with larger misalignments between the input point clouds.

| ![Image](/images/corr_vs_transf.png) | 
|:--:| 
| Correspondence Vs Transformation |