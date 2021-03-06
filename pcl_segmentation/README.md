# EE225B Final Project

**Griffin and Hasith:Targetless Extrinsic Calibration of Camera and LiDAR 
using ML-based segmentation
**

download data at `http://www.cvlibs.net/download.php?file=data_object_velodyne.zip`

kitti2pcl folder : tool described below
lidar_seg.cpp : Region Growing clustering on pcd
pclsegmentation.m : K-means clustering on pcd

## Kitti2pcl
KITTI dataset provides pcl data in bin format and there is a kitti2pcl folder where you can run `cmake` followed by `make` to build a tool to convert bin to standard pcl format. After making, run ./velo2pcd --folder DIR_TO_KITTIDATASET.

## Matlab file
the pclsegmentation.m file implements a similar K-means algorithm(the distance to cluster's mean is parameter instead of number of clusters) for pcl points. However, the quality of result is largely depent on the selection of distance parameter, and it always tends to group things near the lidar together. Maybe another similarity metrics could be tried instead of Euclidean distance but I found another algorithm that generally does a better job. See below.

## Explaination
matlab does not support point cloud processing that well so this LiDAR segmentation is done in C++ using open source Point Cloud Library(pcl). Therefore this code requires at least PCL1.5 to run.
Refer: http://pointclouds.org/

## Algorithm
It uses Machine Learning K nearest search and Region Growing Segmentation algorithm, whose purpose is to merge the points that are close enough in terms of the smoothness constraint. Thereby, the output of this algorithm is the set of clusters, were each cluster is a set of points that are considered to be a part of the same smooth surface. The work of this algorithm is based on the comparison of the angles between the points normals.
See complete comments in the codes for detailed implementation.

## Complilation
The code can be compiled by running `cmake` followed by `make`.

## Results
By running the codes, the outputs in the results can be reproduced.
Don't forget to download the data and put the unzipped training and testing or modify the path in corresponding places.

##Use
```
Usage is: ./lidar_seg pcdfile {-v}

         optional arguments:
           -v visualize segmentation result
