#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace cv;

#define PIXEL_SIZE 0.00000345f
#define POINT_RADIUS 2

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;


vector<Point> pointCloudToPixelCoord(PointCloud::Ptr cloud,
        Mat image, Mat G, Mat P)
{
    vector<Point> pixelCoords;

    for (int i = 0; i < cloud->points.size(); i++)
    {
        int rows = image.rows;
        int cols = image.cols;

        double p_data[4] = {cloud->points[i].x / PIXEL_SIZE,
                            cloud->points[i].y / PIXEL_SIZE,
                            cloud->points[i].z / PIXEL_SIZE,
                            1.0f};

        Mat p = Mat(4, 1, CV_64F, p_data);

        Mat q = P * G * p;
        double z = q.at<double>(2, 0);
        q = q / z;
        
        double u = q.at<double>(0, 0);
        double v = q.at<double>(1, 0);

        if (z > 0 && u > 0 && u <= cols && v > 0 && v <= rows)
        {
            Point marker(u, v);
            pixelCoords.push_back(marker);
        }
    }

    return pixelCoords;
}


int main(int argc, char **argv)
{
    // Read point cloud data
    PointCloud::Ptr cloud(new PointCloud);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ>
            ("../data/lidar_scan_0.pcd", *cloud) == -1)
    {
        cout << "no pointcloud data" << endl;
        return -1;
    }
    
    // Read image data
    Mat image = imread("Acquisition-17391280-0.jpg", 1);
    
    if (!image.data)
    {
        cout << "no image data" << endl;
        return -1;
    }

    // Intrinsic calibration matrix
    double P_data[12] = {1290.5f, 0.0f,    1018.6f, 0.0f,
                         0.0f,    1296.0f, 793.0f,  0.0f,
                         0.0f,    0.0f,    1.0f,    0.0f};
    Mat P = Mat(3, 4, CV_64F, P_data);

    // Extrinsic calibration matrix, initial guess
    double G_data[16] = { 0,  1,  0,  0,
                          0,  0,  1,  0,
                          1,  0,  0,  0,
                          0,  0,  0,  1};
    
    Mat G = Mat(4, 4, CV_64F, G_data);

    vector<Point> pixelCoords = pointCloudToPixelCoord(
        cloud, image, G, P);

    for (int i = 0; i < pixelCoords.size(); i++)
    {
        circle(image, pixelCoords[i], POINT_RADIUS, 
            Scalar(0, 0, 255), FILLED, LINE_8);
    }

    namedWindow("Display Image", WINDOW_NORMAL);
    imshow("Display Image", image);
    waitKey(0);
    
    return 0;
}

