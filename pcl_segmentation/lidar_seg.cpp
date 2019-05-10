#include <iostream>
#include <vector>
#include <string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h> // accelerated decision tree for KNN
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>

int main (int argc, char** argv)
{
  // change the input file name to match input argument
  if (argc < 2)
  {
    std::cout << "please specify input pcd file." << std::endl;
    return (-1);
  }
  // loading the point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> (argv[1], *cloud) == -1)
  {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }
  bool seen = false;
  if (argc == 3)
  {
    if (strcmp(argv[2],"-v") == 0)
    {
      seen = true;
    }
  }
  // loading as tradition, set some paramters
  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);
  
  // regiongrowing function is used here
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  // clusters with less points then minimum will be discarded.
  // setting as an example. This is pararmeters needed
  reg.setMinClusterSize (50);
  // reg.setMaxClusterSize (1000000);
  
  // K nearest neighbor search
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  reg.setInputNormals (normals);
  
  // algorithm parameters here
  reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);
  
  // show seg results
  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);
  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
  std::cout << "First cluster has " << clusters[0].indices.size () << " points." << endl;
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  
  // save reg to file
  std::cout << "Saving segmentation result to seg" << argv[1] << std::endl;
  std::string output = "res_of_";
  output.append(argv[1]);
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZRGB> (output, *colored_cloud, false);
  
  // visualize the segmentation result
  if (seen)
  {
    pcl::visualization::CloudViewer viewer ("Cluster viewer");
    viewer.showCloud(colored_cloud);
    while (!viewer.wasStopped ())
    {
    }
  }
  return (0);
}
