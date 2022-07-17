/*
 * @Author: kinggreat24
 * @Date: 2022-07-05 15:44:06
 * @LastEditTime: 2022-07-09 14:44:30
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /ORB_SLAM2/Thirdparty/monolidar_fusion/examples/monolidar_depth_extraction.cc
 * 可以输入预定的版权声明、个性签名、空行等
 */
// #include <ros/ros.h>
#include "ORBextractor.h"
#include "monolidar_fusion/DepthEstimator.h"
#include "monolidar_fusion/camera_pinhole.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl/filters/conditional_removal.h> //条件滤波
#include <pcl/filters/passthrough.h>

typedef pcl::PointXYZI PointType;

void CalculateFeatureDepthsCurFrame(Mono_Lidar::DepthEstimator &_depthEstimator, const Mono_Lidar::DepthEstimator::Cloud::ConstPtr &cloud_in_cur,
                                    const std::vector<cv::KeyPoint> &kps,
                                    Eigen::VectorXd &depths,
                                    Mono_Lidar::GroundPlane::Ptr &ransacPlane);

Eigen::Matrix<double, 4, 4> toMatrix4d(const cv::Mat &cvMat4);

int ReadPointCloud(const std::string &file, pcl::PointCloud<PointType>::Ptr outpointcloud, bool isBinary);

int main(int argc, char **argv)
{
    //Read camera & lidar extransic parameters
    std::string _path_config_depthEstimator = std::string(argv[1]);
    std::cout << "Read setting file: " << _path_config_depthEstimator << std::endl;
    cv::FileStorage fSettings(_path_config_depthEstimator, cv::FileStorage::READ);

    // std::string _path_config_depthEstimator("");
    // fSettings["config_file"] >> _path_config_depthEstimator;

    Mono_Lidar::DepthEstimator depth_estimator;
    depth_estimator.InitConfig(_path_config_depthEstimator);

    int width = fSettings["Camera.width"];
    int height = fSettings["Camera.height"];
    double fx = fSettings["Camera.fx"];
    double fy = fSettings["Camera.fy"];
    double focal_length = (fx + fy) / 2;
    double principal_point_x = fSettings["Camera.cx"];
    double principal_point_y = fSettings["Camera.cy"];
    std::cout << "Camera parameters: " << std::endl
              << " Camera.fx: " << fx << "  Camera.fy: " << fx << " Camera.cx: " << principal_point_x << " Camera.cy: " << principal_point_y << std::endl;
    std::shared_ptr<CameraPinhole> pCamPinhole(new CameraPinhole(width, height, focal_length, principal_point_x, principal_point_y));

    // 激光与相机之间的外参关系
    cv::Mat lidarCameraExtriParam = cv::Mat::zeros(4, 4, CV_32F);
    cv::FileNode LidarCameraExtrinParamNode = fSettings["LidarCamera.extrinsicParameters"];
    if (LidarCameraExtrinParamNode.type() != cv::FileNode::SEQ)
    {
        std::cerr << "LidarCamera.extrinsicParameters is not a sequence!" << std::endl;
        return 0;
    }
    cv::FileNodeIterator it = LidarCameraExtrinParamNode.begin(), it_end = LidarCameraExtrinParamNode.end();
    for (int count = 0; it != it_end; it++, count++)
    {
        int row = count / 4;
        int col = count % 4;
        lidarCameraExtriParam.at<float>(row, col) = *it;
    }
    int need_inverse = fSettings["LidarCamera.need_inverse"];
    Eigen::Matrix4d Tcl = Eigen::Matrix4d::Identity();
    if (!need_inverse)
        Tcl = toMatrix4d(lidarCameraExtriParam);
    else
        Tcl = toMatrix4d(lidarCameraExtriParam).inverse();

    Eigen::Affine3d T_cam_lidar(Tcl);
    depth_estimator.Initialize(pCamPinhole, T_cam_lidar);

    // GroundPlane::Ptr pGroundPlane(new GroundPlane());
    Mono_Lidar::DepthEstimatorParameters depth_estimator_parameters_;
    depth_estimator_parameters_.fromFile(_path_config_depthEstimator);
    double plane_inlier_threshold = depth_estimator_parameters_.ransac_plane_refinement_treshold;

    // Mono_Lidar::GroundPlane::Ptr gp =
    //     std::make_shared<Mono_Lidar::SemanticPlane>(img_ptr->image, pCamPinhole, gp_labels, plane_inlier_threshold);

    // Init plane ransac
    Mono_Lidar::GroundPlane::Ptr gp = std::make_shared<Mono_Lidar::RansacPlane>(
        std::make_shared<Mono_Lidar::DepthEstimatorParameters>(depth_estimator_parameters_));

    //读取图像与激光数据
    std::string image_name(""), lidar_name("");
    fSettings["image_name"] >> image_name;
    fSettings["lidar_name"] >> lidar_name;
    cv::Mat imgray = cv::imread(image_name, CV_LOAD_IMAGE_GRAYSCALE);
    pcl::PointCloud<PointType>::Ptr lidar_pc(new pcl::PointCloud<PointType>());
    ReadPointCloud(lidar_name, lidar_pc, true);

    // pcl::PassThrough<pcl::PointXYZI> pass_through;
    // pass_through.setInputCloud(lidar_pc);
    // pass_through.setFilterFieldName("z");
    // float sensor_height = 1.5;
    // float minDepth = -1.0 * sensor_height - 0.8;
    // float maxDepth = -1.0 * sensor_height + 0.8;
    // pass_through.setFilterLimits(minDepth, maxDepth); //-0.25<x<0.15 为内点
    // pcl::PointCloud<PointType>::Ptr cloud_after_Condition(new pcl::PointCloud<PointType>());
    // pass_through.filter(*cloud_after_Condition);

    // cloud_after_Condition->height = 1;
    // cloud_after_Condition->width = cloud_after_Condition->size();
    // pcl::io::savePCDFileASCII("./test_groud.pcd", *cloud_after_Condition); //将点云保存到PCD文件中

    //特征提取
    ORB_SLAM2::ORBextractor *mpOrbExtractor = new ORB_SLAM2::ORBextractor(3000, 1.2, 8, 20, 7);
    std::vector<cv::KeyPoint> vKeyFeatures;
    cv::Mat mDescriptor;
    Eigen::VectorXd depths;
    (*mpOrbExtractor)(imgray, cv::Mat(), vKeyFeatures, mDescriptor);

    int N = vKeyFeatures.size();
    std::cout << "Extract orb feature size: " << N << std::endl;

    CalculateFeatureDepthsCurFrame(depth_estimator, lidar_pc, vKeyFeatures, depths, gp);

    std::cout << "feature depth extraction completed" << std::endl;

    // 显示深度拟合的结果
    cv::Mat imColor = imgray.clone();
    if (imColor.channels() == 1)
        cv::cvtColor(imColor, imColor, CV_GRAY2BGR);
    const float r = 5.0;
    float v_min = 5.0;
    float v_max = 80.0;
    float dv = v_max - v_min;
    for (size_t i = 0; i < vKeyFeatures.size(); i++)
    {
        cv::Point2f pt1, pt2;
        pt1.x = vKeyFeatures[i].pt.x - r;
        pt1.y = vKeyFeatures[i].pt.y - r;
        pt2.x = vKeyFeatures[i].pt.x + r;
        pt2.y = vKeyFeatures[i].pt.y + r;

        float v = depths[i];
        if (v <= 0)
            continue;

        float r = 1.0;
        float g = 1.0;
        float b = 1.0;
        if (v < v_min)
            v = v_min;
        if (v > v_max)
            v = v_max;

        if (v < v_min + 0.25 * dv)
        {
            r = 0.0;
            g = 4 * (v - v_min) / dv;
        }
        else if (v < (v_min + 0.5 * dv))
        {
            r = 0.0;
            b = 1 + 4 * (v_min + 0.25 * dv - v) / dv;
        }
        else if (v < (v_min + 0.75 * dv))
        {
            r = 4 * (v - v_min - 0.5 * dv) / dv;
            b = 0.0;
        }
        else
        {
            g = 1 + 4 * (v_min + 0.75 * dv - v) / dv;
            b = 0.0;
        }

        cv::rectangle(imColor, pt1, pt2, 255 * cv::Scalar(r, g, b));
        cv::circle(imColor, vKeyFeatures[i].pt, 2, 255 * cv::Scalar(r, g, b), -1);
    }

    cv::imshow("im_color", imColor);
    cv::imwrite("/home/kinggreat24/pc/limo_depth_pred.png", imColor);
    cv::waitKey(0);

    // depth_estimator.CalculateDepth();
    return 0;
}

void CalculateFeatureDepthsCurFrame(Mono_Lidar::DepthEstimator &_depthEstimator, const Mono_Lidar::DepthEstimator::Cloud::ConstPtr &cloud_in_cur,
                                    const std::vector<cv::KeyPoint> &kps,
                                    Eigen::VectorXd &depths,
                                    Mono_Lidar::GroundPlane::Ptr &ransacPlane)
{
    // Convert the feature points to the interface format for the DepthEstimator
    int frameCount = kps.size();
    depths.resize(frameCount);
    Eigen::Matrix2Xd featureCoordinates(2, frameCount);

    int i = 0;
    for (const auto &kp : kps)
    {
        // insert features of the current frame
        featureCoordinates(0, i) = kp.pt.x;
        featureCoordinates(1, i) = kp.pt.y;
        i++;
    }

    std::cout << "*******       CalculateDepth       *********" << std::endl;
    _depthEstimator.CalculateDepth(cloud_in_cur, featureCoordinates, depths, ransacPlane);
}

Eigen::Matrix<double, 4, 4> toMatrix4d(const cv::Mat &cvMat4)
{
    Eigen::Matrix<double, 4, 4> M;

    M << cvMat4.at<float>(0, 0), cvMat4.at<float>(0, 1), cvMat4.at<float>(0, 2), cvMat4.at<float>(0, 3),
        cvMat4.at<float>(1, 0), cvMat4.at<float>(1, 1), cvMat4.at<float>(1, 2), cvMat4.at<float>(1, 3),
        cvMat4.at<float>(2, 0), cvMat4.at<float>(2, 1), cvMat4.at<float>(2, 2), cvMat4.at<float>(2, 3),
        cvMat4.at<float>(3, 0), cvMat4.at<float>(3, 1), cvMat4.at<float>(3, 2), cvMat4.at<float>(3, 3);

    return M;
}

int ReadPointCloud(const std::string &file, pcl::PointCloud<PointType>::Ptr outpointcloud, bool isBinary)
{
    // pcl::PointCloud<PointType>::Ptr curPointCloud(new pcl::PointCloud<PointType>());
    if (isBinary)
    {
        // load point cloud
        std::fstream input(file.c_str(), std::ios::in | std::ios::binary);
        if (!input.good())
        {
            std::cerr << "Could not read file: " << file << std::endl;
            exit(EXIT_FAILURE);
        }
        //LOG(INFO)<<"Read: "<<file<<std::endl;

        for (int i = 0; input.good() && !input.eof(); i++)
        {
            pcl::PointXYZI point;
            input.read((char *)&point.x, 3 * sizeof(float));
            input.read((char *)&point.intensity, sizeof(float));

            //remove all points behind image plane (approximation)
            /*if (point.x < mMinDepth)
                continue;*/
            float dist = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (dist < 2)
                continue;

            outpointcloud->points.push_back(point);
        }
    }
    else
    {
        if (-1 == pcl::io::loadPCDFile<pcl::PointXYZI>(file, *outpointcloud))
        {
            std::cerr << "Could not read file: " << file << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    outpointcloud->height = 1;
    outpointcloud->width = outpointcloud->points.size();

    // SavePointCloudPly("/home/bingo/pc/loam/pointcloud/pc.ply",outpointcloud);
    return outpointcloud->points.size();
}