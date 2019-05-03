# EE225B Final Project

**Griffin and Hasith:Targetless Extrinsic Calibration of Camera and LiDAR 
using ML-based segmentation
**

## Explaination
matlab does not support point cloud processing that well so this LiDAR segmentation is done in C++ using open source Point Cloud Library(pcl). Therefore this code requires at least PCL1.5 to run.
Refer: http://pointclouds.org/

## Algorithm
It uses Machine Learning K nearest search and Region Growing Segmentation algorithm, whose purpose is to merge the points that are close enough in terms of the smoothness constraint. Thereby, the output of this algorithm is the set of clusters, were each cluster is a set of points that are considered to be a part of the same smooth surface. The work of this algorithm is based on the comparison of the angles between the points normals.
See complete comments in the codes for detailed implementation.

## Complilation
The code can be compiled by running `cmake` followed by `make`.

##Use
```
Usage is: ./lidar_seg pcdfile {-v}

         optional arguments:
           -v visualize segmentation result
