/*
 * @Author: your name
 * @Date: 2021-09-16 21:21:47
 * @LastEditTime: 2021-09-28 19:52:16
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /NLS_examples/hybrid_visual_odometry.cpp
 */

#include <ros/ros.h>

#include "utils.h"
#include "Frame.h"

#include "lidar_sparse_align/SparseLidarAlign.h"
#include "lidar_sparse_align/feature_align.h"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include <opencv2/core/eigen.hpp>

// global variables
Eigen::Matrix4d cameraLidarExtrinsic = Eigen::Matrix4d::Identity();
vk::PinholeCamera *pPinhole_camera = static_cast<vk::PinholeCamera *>(NULL);
std::vector<Eigen::Matrix4d> global_gt_poses_;

ros::Publisher pub_image_, pub_disparity_, pub_path_, pub_odometry_;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "hybrid_vlo");
    ros::NodeHandle nh, nh_private("~");

    pub_image_     = nh.advertise<sensor_msgs::Image>("image_left",1);
    pub_disparity_ =  nh.advertise<sensor_msgs::Image>("image_left",1);
    // pub_path_ = 


    std::string strSettingPath("");
    nh_private.param("strSettingPath",strSettingPath,strSettingPath);
    if(strSettingPath.empty())
    {
        ROS_ERROR("Empty setting file, please check");
        return -1;
    }

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    int width = fSettings["Camera.width"];
    int height = fSettings["Camera.height"];
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    pPinhole_camera = new vk::PinholeCamera(width, height, fx, fy, cx, cy);

    // Retrieve paths to images
    std::vector<std::string> vstrImageLeft;
    std::vector<std::string> vstrImageRight;
    std::vector<std::string> vstrLidar;
    std::vector<double> vTimestamps;
    std::string data_sequene = (std::string)fSettings["data_sequence"];
    hybrid_vlo::LoadImages(data_sequene, vstrImageLeft, vstrImageRight, vstrLidar, vTimestamps);
    const int nImages = vstrImageLeft.size();
    std::cout << "Load images: " << nImages << std::endl;

    // 读取相机的真实姿态
    hybrid_vlo::LoadGroundTruth(data_sequene + "/00.txt", global_gt_poses_);

    // 相机与激光雷达外参
    cv::Mat T;
    fSettings["extrinsicMatrix"] >> T;
    cv::cv2eigen(T, cameraLidarExtrinsic);

    std::cout << "camera lidar extrinsicMatrix: " << std::endl
              << cameraLidarExtrinsic << std::endl;

    //
    std::cout << "construct ORB feature textractor" << std::endl;
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float scale_factor = fSettings["ORBextractor.scaleFactor"];
    int nlevels = fSettings["ORBextractor.nLevels"];
    int iniThFAST = fSettings["ORBextractor.iniThFAST"];
    int minThFAST = fSettings["ORBextractor.minThFAST"];
    ORB_SLAM2::ORBextractor *pOrbExtractor = new ORB_SLAM2::ORBextractor(nFeatures, scale_factor, nlevels, iniThFAST, minThFAST);

    // 双目基线长度
    float bf = fSettings["Stereo.bf"];
    std::cout << "stereo baseline: " << bf << std::endl;

    // tracking levels
    int nLevels = fSettings["Tracker.levels"];

    // 视觉激光雷达直接法
    hybrid_vlo::_tracker_t tracker_info;
    tracker_info.levels = fSettings["Tracker.levels"];
    tracker_info.max_iteration = fSettings["Tracker.max_iteration"];
    tracker_info.max_level = fSettings["Tracker.max_level"];
    tracker_info.min_level = fSettings["Tracker.min_level"];
    tracker_info.use_weight_scale = true;
    tracker_info.scale_estimator = (std::string)fSettings["Tracker.scale_estimator"];
    tracker_info.weight_function = (std::string)fSettings["Tracker.weight_function"];
    tracker_info.print();
    hybrid_vlo::SparseLidarAlign *mpLidarSparseAlign = new hybrid_vlo::SparseLidarAlign(pPinhole_camera, tracker_info);

    // 间接法
    hybrid_vlo::_tracker_t feature_tracker_info;
    feature_tracker_info.scale_estimator = (std::string)fSettings["FTracker.scale_estimator"];
    feature_tracker_info.weight_function = (std::string)fSettings["FTracker.weight_function"];
    feature_tracker_info.set_scale_estimator_type();
    feature_tracker_info.set_weight_function_type();
    hybrid_vlo::FeatureAlign *mpFeatureAlign = new hybrid_vlo::FeatureAlign(pPinhole_camera, feature_tracker_info);
    mpFeatureAlign->verbose_ = true;


    // 构造当前帧与参考帧
    std::cout << "construct image frame" << std::endl;
    int src_idx = 0, tar_idx = 1;
    cv::Mat img_last_L = cv::imread(vstrImageLeft[src_idx], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img_last_R = cv::imread(vstrImageRight[src_idx], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img_cur_L = cv::imread(vstrImageLeft[tar_idx], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img_cur_R = cv::imread(vstrImageRight[tar_idx], CV_LOAD_IMAGE_UNCHANGED);
    hybrid_vlo::Frame *pLastFrame = new hybrid_vlo::Frame(img_last_L, img_last_R, vstrLidar[src_idx],
                                                          pOrbExtractor, pPinhole_camera, nLevels, cameraLidarExtrinsic, bf);
    hybrid_vlo::Frame *pCurFrame = new hybrid_vlo::Frame(img_cur_L, img_cur_R, vstrLidar[tar_idx],
                                                         pOrbExtractor, pPinhole_camera, nLevels, cameraLidarExtrinsic, bf);

    // 直接法跟踪
    Eigen::Matrix4d Twc_last = global_gt_poses_.at(src_idx);
    Eigen::Matrix4d Twc_cur = global_gt_poses_.at(tar_idx);
    Eigen::Matrix4d Tcl_gt = Twc_cur.inverse() * Twc_last;

    std::cout << "Tcl_gt: " << std::endl << Tcl_gt.matrix() << std::endl;

    std::vector<cv::Point2f> prepoints, nextpoints;
    std::vector<uchar> track_state;
    cv::Mat track_results;
    hybrid_vlo::klt_feature_tracking(img_last_L, img_cur_L, pLastFrame->mvKeys, prepoints, nextpoints, track_state, cv::Mat(), track_results, true);
    
    // 显示特征跟踪的结果
    // track_results = hybrid_vlo::showTrackingResults(img_last_L, img_cur_L, prepoints, nextpoints, track_state);
    // cv::imshow("klt_tracking", track_results);
    // cv::waitKey(0);

    Sophus::SE3 Tcl_gn_feature(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    mpFeatureAlign->tracking(prepoints, pLastFrame->mvDepths, nextpoints, track_state, Tcl_gn_feature);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout<<"feature tracking time: "<<ttrack<<std::endl;
     

    for(int ni=0;ni<nImages;ni++)
    {
        
    }



    // 视觉激光融合直接法跟踪 
    Sophus::SE3 Tcl_gn(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    mpLidarSparseAlign->tracking(pLastFrame, pCurFrame, Tcl_gn);
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double tlidartrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
    std::cout<<"direct tracking time: "<<tlidartrack<<std::endl;
    std::cout << "lidar sparse tracking result: " << std::endl
              << Tcl_gn.matrix() << std::endl;

    return 0;
}