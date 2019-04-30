#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/impl/sift_keypoint.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

// return a point cloud with only the keypoints
pcl::PointCloud<pcl::PointXYZ>::Ptr sift_keypoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud)
{
    // SIFT parameters
    const float min_scale = 0.01f;
    const int n_octaves = 3;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.001f;
    const float search_radius = 0.04f;

    // estimate normals of point cloud
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(
        new pcl::PointCloud<pcl::PointNormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(
        new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setInputCloud(input_cloud);
    ne.setSearchMethod(tree_n);
    ne.setRadiusSearch(search_radius);
    ne.compute(*cloud_normals);

    // copy points to normals
    for (size_t i = 0; i < cloud_normals->points.size(); i++)
    {
        cloud_normals->points[i].x = input_cloud->points[i].x;
        cloud_normals->points[i].y = input_cloud->points[i].y;
        cloud_normals->points[i].z = input_cloud->points[i].z;
    }
    
    // Estimate SIFT points with normals
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree(
        new pcl::search::KdTree<pcl::PointNormal>());
    
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_normals);
    sift.compute(result);

    cout << "Number of sift keypoints: " << result.points.size()
        << endl;

    // visualization
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(
        new pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(result, *keypoints);
    
    return keypoints;
}

int main(int argc, char **argv)
{
    // Read point cloud data
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ>
            ("../data/lidar_scan_0.pcd", *cloud) == -1)
    {
        cout << "no pointcloud data" << endl;
        return -1;
    }

    cout << "Read point cloud with " << cloud->points.size()
        << " points" << endl;

    /*
    // SIFT parameters
    const float min_scale = 0.01f;
    const int n_octaves = 3;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.001f;

    // estimate normals of point cloud
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(
        new pcl::PointCloud<pcl::PointNormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(
        new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree_n);
    ne.setRadiusSearch(0.04);
    ne.compute(*cloud_normals);

    // copy points to normals
    for (size_t i = 0; i < cloud_normals->points.size(); i++)
    {
        cloud_normals->points[i].x = cloud->points[i].x;
        cloud_normals->points[i].y = cloud->points[i].y;
        cloud_normals->points[i].z = cloud->points[i].z;
    }
    
    // Estimate SIFT points with normals
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree(
        new pcl::search::KdTree<pcl::PointNormal>());
    
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_normals);
    sift.compute(result);

    cout << "Number of sift keypoints: " << result.points.size()
        << endl;

    // visualization
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(
        new pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(result, *keypoints);
    */

    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints = 
        sift_keypoints(cloud);

    pcl::visualization::CloudViewer key_viewer("keypoint viewer");
    key_viewer.showCloud(keypoints);
    while (!key_viewer.wasStopped())
    {
    }

    return 0;
}


