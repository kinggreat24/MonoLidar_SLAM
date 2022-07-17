/*
 * @Author: your name
 * @Date: 2021-09-20 15:08:15
 * @LastEditTime: 2022-07-12 10:02:45
 * @LastEditors: kinggreat24
 * @Description: In User Settings Edit
 * @FilePath: /ORB_SLAM2/Examples/Monocular-Lidar/hybrid_track_ws/src/hybrid_vlo/include/hybrid_vlo/utils.h
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>
#include <iomanip>
#include <chrono>

#include <Eigen/Core>

#include <opencv2/core.hpp>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl-1.7/pcl/io/pcd_io.h>

#include <vikit/pinhole_camera.h>

#include "lidar_sparse_align/WeightFunction.h"


// CUDA
#include <cuda_runtime.h>

// SGM
#include <libsgm.h>

namespace hybrid_vlo
{
    typedef pcl::PointXYZI PointType;

    typedef Eigen::Matrix<float, 2, 1> Vector2;
    typedef Eigen::Matrix<double, 4, 1> Vector4d;
    typedef Eigen::Matrix<float, 6, 1> Vector6;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<float, 2, 6> Matrix2x6;
    typedef Eigen::Matrix<double, 4, 4> Matrix4d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    // *******************              读取数据            ********************
    void LoadGroundTruth(const std::string &gt_file, std::vector<Eigen::Matrix<double, 4, 4>> &gtPoses);

    void LoadImages(const std::string &strPathToSequence, std::vector<std::string> &vstrImageLeft,
                    std::vector<std::string> &vstrImageRight, vector<string> &vstrImageDisparity, std::vector<std::string> &vstrLidar, std::vector<double> &vTimestamps);

    int ReadPointCloud(const std::string &file, pcl::PointCloud<pcl::PointXYZI>::Ptr outpointcloud, bool isBinary);

    // *******************              图像金字塔            ********************

    template <typename T>
    static void pyrDownMeanSmooth(const cv::Mat &in, cv::Mat &out);

    void create_image_pyramid(const cv::Mat &img_level_0, int n_levels, std::vector<cv::Mat> &pyramid);


    // 特征追踪
    int klt_feature_tracking(const cv::Mat &imgrayLast, const cv::Mat &imgrayCur, const std::vector<cv::KeyPoint> &last_keypoints, std::vector<cv::Point2f> &prevpoints, std::vector<cv::Point2f> &nextpoints,
                            std::vector<uchar>& tracked_states, const cv::Mat &dynamic_mask, cv::Mat& debug_image, bool flow_back);


    cv::Mat showTrackingResults(const cv::Mat& imgrayLast, const cv::Mat& imgrayCur, const std::vector<cv::Point2f> &last_keypoints, std::vector<cv::Point2f> &prevpoints, std::vector<uchar> &tracked_states);

    // *******************              显示函数            ********************
    void ShowPointClouds(const pcl::PointCloud<pcl::PointXYZI>::Ptr &mpLidarPointCloud, const std::vector<cv::Mat> &mvImgPyramid,
                         const vk::PinholeCamera *mpPinholeCamera, cv::Mat &image_out, size_t num_level);

    Eigen::Vector3d get_false_color(float depth, const float min_depth, const float max_depth);

    typedef struct _tracker_t tracker_t;
    struct _tracker_t
    {
        // 金字塔层数以及最大迭代次数
        int levels;
        int min_level;
        int max_level;
        int max_iteration;

        // 直接法权重函数
        bool use_weight_scale = true;
        std::string scale_estimator;
        std::string weight_function;
        ScaleEstimatorType scale_estimator_type;
        WeightFunctionType weight_function_type;

        void print()
        {
            std::cout << "levels: " << levels << std::endl
                      << "min_level: " << min_level << std::endl
                      << "max_level: " << max_level << std::endl
                      << "max_iteration: " << max_iteration << std::endl
                      << "scale_estimator: " << scale_estimator << std::endl
                      << "weight_function_type: " << weight_function << std::endl;
        }

        /****************             权重函数            ******************/
        void set_scale_estimator_type()
        {
            if (!scale_estimator.compare("None"))
                use_weight_scale = false;

            if (!scale_estimator.compare("TDistributionScale"))
                scale_estimator_type = ScaleEstimatorType::TDistributionScale;

            if (!scale_estimator.compare("MADScale"))
                scale_estimator_type = ScaleEstimatorType::MADScale;

            if (!scale_estimator.compare("NormalDistributionScale"))
                scale_estimator_type = ScaleEstimatorType::NormalDistributionScale;

            cerr << "ScaleType : " << static_cast<int>(scale_estimator_type);
        }

        void set_weight_function_type()
        {
            if (!weight_function.compare("TDistributionWeight"))
                weight_function_type = WeightFunctionType::TDistributionWeight;

            if (!weight_function.compare("HuberWeight"))
                weight_function_type = WeightFunctionType::HuberWeight;

            if (!weight_function.compare("TukeyWeight"))
                weight_function_type = WeightFunctionType::TukeyWeight;

            cerr << "Weight function : " << static_cast<int>(weight_function_type);
        }
    };


    typedef struct _feature_tracker_t feature_tracker_t;
    struct _feature_tracker_t
    {
        // 最大迭代次数
        int max_iteration;

        // 直接法权重函数
        bool use_weight_scale = true;
        std::string scale_estimator;
        std::string weight_function;
        ScaleEstimatorType scale_estimator_type;
        WeightFunctionType weight_function_type;

        void print()
        {
            std::cout << "max_iteration: "        << max_iteration << std::endl
                      << "scale_estimator: "      << scale_estimator << std::endl
                      << "weight_function_type: " << weight_function << std::endl;
        }

        /****************             权重函数            ******************/
        void set_scale_estimator_type()
        {
            if (!scale_estimator.compare("None"))
                use_weight_scale = false;

            if (!scale_estimator.compare("TDistributionScale"))
                scale_estimator_type = ScaleEstimatorType::TDistributionScale;

            if (!scale_estimator.compare("MADScale"))
                scale_estimator_type = ScaleEstimatorType::MADScale;

            if (!scale_estimator.compare("NormalDistributionScale"))
                scale_estimator_type = ScaleEstimatorType::NormalDistributionScale;

            cerr << "ScaleType : " << static_cast<int>(scale_estimator_type);
        }

        void set_weight_function_type()
        {
            if (!weight_function.compare("TDistributionWeight"))
                weight_function_type = WeightFunctionType::TDistributionWeight;

            if (!weight_function.compare("HuberWeight"))
                weight_function_type = WeightFunctionType::HuberWeight;

            if (!weight_function.compare("TukeyWeight"))
                weight_function_type = WeightFunctionType::TukeyWeight;

            cerr << "Weight function : " << static_cast<int>(weight_function_type);
        }
    };





    // 双目视觉相关
    struct device_buffer
    {
        device_buffer() : data(nullptr) {}
        device_buffer(size_t count) { allocate(count); }
        void allocate(size_t count) { cudaMalloc(&data, count); }
        ~device_buffer() { cudaFree(data); }
        void *data;
    };

    // Camera Parameters
    struct CameraParameters
    {
        float fu;       //!< focal length x (pixel)
        float fv;       //!< focal length y (pixel)
        float u0;       //!< principal point x (pixel)
        float v0;       //!< principal point y (pixel)
        float baseline; //!< baseline (meter)
        float height;   //!< height position (meter), ignored when ROAD_ESTIMATION_AUTO
        float tilt;     //!< tilt angle (radian), ignored when ROAD_ESTIMATION_AUTO
    };

    // Transformation between pixel coordinate and world coordinate
    struct CoordinateTransform
    {
        CoordinateTransform(const CameraParameters &camera) : camera(camera)
        {
            sinTilt = (sinf(camera.tilt));
            cosTilt = (cosf(camera.tilt));
            bf = camera.baseline * camera.fu;
            invfu = 1.f / camera.fu;
            invfv = 1.f / camera.fv;
        }

        inline cv::Point3f imageToWorld(const cv::Point2f &pt, float d) const
        {
            const float u = pt.x;
            const float v = pt.y;

            const float Zc = bf / d;
            const float Xc = invfu * (u - camera.u0) * Zc;
            const float Yc = invfv * (v - camera.v0) * Zc;

            const float Xw = Xc;
            const float Yw = Yc * cosTilt + Zc * sinTilt;
            const float Zw = Zc * cosTilt - Yc * sinTilt;

            return cv::Point3f(Xw, Yw, Zw);
        }

        CameraParameters camera;
        float sinTilt, cosTilt, bf, invfu, invfv;
    };
}
#endif //UTILS_H_