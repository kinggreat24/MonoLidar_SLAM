/*
 * @Author: your name
 * @Date: 2021-09-20 20:18:41
 * @LastEditTime: 2021-09-29 17:44:32
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /NLS_examples/include/Frame.h
 */


#ifndef FRAME_H_
#define FRAME_H_

#include "utils.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "ORBextractor.h"

//三角网
#include "include_fade25d/Fade_2D.h"

namespace hybrid_vlo{
    


class Frame
{
public:
    Frame(cv::Mat &imL, cv::Mat &imR, std::string& lidar_file, ORB_SLAM2::ORBextractor* pORBExtractor, vk::PinholeCamera* pinhole_cam, int nlevles, 
        Eigen::Matrix4d& lidar2camera_matrix, float bf, bool depth_flag = false);
    
    // Frame(const Frame &frame);

    ~Frame();
    
    void ExtractORB(cv::Mat& imGray);
    void PointSampling();

    cv::Mat ComputeStereoDisparity(const cv::Mat &left_image, const cv::Mat &right_image,
                               const int disp_size = 128, const bool subpixel = 1,
                               const int output_depth = 16);
    void Compute3dKeyPoints(const std::vector<cv::Point2f> &vKeyPoints, const cv::Mat &disparity, const CameraParameters &camera, std::vector<cv::Point3f> &pt_3ds);

    void ComputeStereoPointCloud(const cv::Mat &image_gray, const cv::Mat &disparity, const CameraParameters &camera, bool subpixeled,
                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_3d_ptr);

    int ComputeStereoDepth(const cv::Mat &disparity, const float bf);

    inline static void jacobian_xyz2uv(const Eigen::Vector3f& xyz_in_f, Matrix2x6& J)
    {
        const float x = xyz_in_f[0];
        const float y = xyz_in_f[1];
        const float z_inv = 1./xyz_in_f[2];
        const float z_inv_2 = z_inv*z_inv;

        J(0,0) = -z_inv;              // -1/z
        J(0,1) = 0.0;                 // 0
        J(0,2) = x*z_inv_2;           // x/z^2
        J(0,3) = y*J(0,2);            // x*y/z^2
        J(0,4) = -(1.0 + x*J(0,2));   // -(1.0 + x^2/z^2)
        J(0,5) = y*z_inv;             // y/z

        J(1,0) = 0.0;                 // 0
        J(1,1) = -z_inv;              // -1/z
        J(1,2) = y*z_inv_2;           // y/z^2
        J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
        J(1,4) = -J(0,3);             // -x*y/z^2
        J(1,5) = -x*z_inv;            // x/z
    }

    pcl::PointCloud<PointType>::Ptr GetMagPointCloud(){return mpMagLidarPointCloud;}
    std::vector<cv::Mat> GetImgPyramid(){return mvImgPyramid_;}


    // 特征深度估计
    GEOM_FADE25D::Fade_2D *CreateTerrain(pcl::PointCloud<PointType>::Ptr &pc_cam);
    // GEOM_FADE25D::Fade_2D *CreateTerrain(std::unordered_map<uint16_t, depth_clustering::Cloud> &clusters);
    float DepthFitting(GEOM_FADE25D::Fade_2D *pdt, const cv::Point2f &pt);
    void DrawObjectDenuary(GEOM_FADE25D::Fade_2D *pdt_all,
        const std::vector<cv::KeyPoint> &mvKeys, std::string file_name);

    void ComputeImageBounds(const cv::Mat &imLeft);


    // 显示函数
    void ShowPointClouds(const pcl::PointCloud<pcl::PointXYZI>::Ptr &mpLidarPointCloud,
                                cv::Mat &image_out, size_t num_level);

    void ShowFeaturePoints(cv::Mat &image_out);

    cv::Scalar randomColor(int64 seed);
public:
    static long unsigned int nNextId;
    int mnId;
   
    int N;

    std::vector<cv::KeyPoint> mvKeys;
    std::vector<float> mvDepths;
    cv::Mat mDescriptor;

    std::vector<cv::Mat> mvImgPyramid_;
    vk::PinholeCamera* mpPinholeCam_;

    pcl::PointCloud<PointType>::Ptr mpMagLidarPointCloud;
    pcl::PointCloud<PointType>::Ptr mpLidarPointCloudCamera;


    cv::Mat mDisparityImage;
    pcl::PointCloud<PointType>::Ptr mpStereoPointCloudCamera;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

private:    
    ORB_SLAM2::ORBextractor* mpORBExtractor;
    Eigen::Matrix4d mLidar2camMatrix;
};


}



#endif//FRAME_H_