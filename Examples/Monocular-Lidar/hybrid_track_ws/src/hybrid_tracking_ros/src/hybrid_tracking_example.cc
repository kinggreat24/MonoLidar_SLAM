/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <opencv2/core/core.hpp>

#include "Converter.h"

#include <pcl-1.7/pcl/point_types.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl-1.7/pcl/common/transforms.h>

#include <System.h>
#include "lidar_sparse_align/HybridSparseAlign.h"
#include "lidar_sparse_align/SparseLidarAlign.h"
#include "lidar_sparse_align/SparseReprojAlign.h"
#include "ORBmatcher.h"
#include "Converter.h"

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>

//cv_bridge
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <geometry_msgs/TransformStamped.h>

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<string> &vstrLidarFilenames, vector<double> &vTimestamps);
void DrawFeatureMatching(const ORB_SLAM2::Frame &frame1, const ORB_SLAM2::Frame &frame2, std::vector<int> &matches);
void SaveTrajectoryKITTI(const string &filename, const std::vector<Eigen::Matrix4d> &vOdomPoses);

ros::Publisher pub_odom_path, pub_frame_img;

nav_msgs::Path lidar_odom_path;

//Camera 2 world coordinate
Eigen::Matrix3d R_transform; // camera_link 2 base_link
Eigen::Quaterniond q_transform;

void PublishTF(Sophus::SE3 Twc);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "hybrid_tracking_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    std::string setting_file(""), voc_file(""), data_sequence("");
    nh_private.param<std::string>("setting_file", setting_file, "");
    nh_private.param<std::string>("voc_file", voc_file, "");
    nh_private.param<std::string>("data_sequence", data_sequence, "");

    // setup ros environment
    pub_odom_path = nh.advertise<nav_msgs::Path>("hybrid_tracking_odometry_path", 1);
    pub_frame_img = nh.advertise<sensor_msgs::Image>("current_frame", 1);

    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    q_transform = Eigen::Quaterniond(R_transform);

    // Load camera parameters from settings file
    cv::FileStorage fSettings(setting_file, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    // K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    // DistCoef.copyTo(mDistCoef);

    int cols = fSettings["Camera.width"];
    int rows = fSettings["Camera.height"];

    vk::PinholeCamera *mpPinholeCamera = static_cast<vk::PinholeCamera *>(NULL);
    if (k3 != 0)
        mpPinholeCamera = new vk::PinholeCamera(cols, rows, fx, fy, cx, cy,
                                                DistCoef.at<float>(0), DistCoef.at<float>(1), DistCoef.at<float>(2), DistCoef.at<float>(3), DistCoef.at<float>(4));
    else
        mpPinholeCamera = new vk::PinholeCamera(cols, rows, fx, fy, cx, cy,
                                                DistCoef.at<float>(0), DistCoef.at<float>(1), DistCoef.at<float>(2), DistCoef.at<float>(3));

    float mbf = fSettings["Camera.bf"];
    float mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
    cout << endl
         << "Depth Threshold (Close/Far Points): " << mThDepth << endl;

    //Load ORB Vocabulary
    cout << endl
         << "Loading ORB Vocabulary. This could take a while..." << endl;

    ORB_SLAM2::ORBVocabulary *mpVocabulary = new ORB_SLAM2::ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromBinFile(voc_file + ".bin");
    if (!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << voc_file << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl
         << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
    ORB_SLAM2::ORBextractor *mpORBextractorLeft = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // 相机与激光雷达之间的外参
    cv::Mat lidarCameraExtriParam = cv::Mat::zeros(4, 4, CV_32F);
    cv::FileNode LidarCameraExtrinParamNode = fSettings["LidarCamera.extrinsicParameters"];
    if (LidarCameraExtrinParamNode.type() != cv::FileNode::SEQ)
    {
        std::cerr << "LidarCamera.extrinsicParameters is not a sequence!" << std::endl;
    }
    cv::FileNodeIterator it = LidarCameraExtrinParamNode.begin(), it_end = LidarCameraExtrinParamNode.end();
    for (int count = 0; it != it_end; it++, count++)
    {
        int row = count / 4;
        int col = count % 4;
        lidarCameraExtriParam.at<float>(row, col) = *it;
    }
    int need_inverse = fSettings["LidarCamera.need_inverse"];
    Eigen::Matrix4d Tcam_ldiar = Eigen::Matrix4d::Identity();
    if (!need_inverse)
        Tcam_ldiar = ORB_SLAM2::Converter::toMatrix4d(lidarCameraExtriParam);
    else
        Tcam_ldiar = ORB_SLAM2::Converter::toMatrix4d(lidarCameraExtriParam).inverse();

    // 特征深度提取
    std::string depth_init_config("");
    fSettings["depth_init_config"] >> depth_init_config;
    std::cout << "Init LIMO feature depth initialization" << std::endl;
    int width = fSettings["Camera.width"];
    int height = fSettings["Camera.height"];
    int depth_type = fSettings["depth_init_type"];
    ORB_SLAM2::eDepthInitMethod depth_init_type = ORB_SLAM2::eDepthInitMethod(depth_type);

    // 激光雷达的类型
    std::string lidar_type("");
    fSettings["Lidar.type"] >> lidar_type;
    ORB_SLAM2::eLidarSensorType lidar_sensor_type;
    if ("VLP_16" == lidar_type)
        lidar_sensor_type = ORB_SLAM2::VLP_16;
    else if ("HDL_32" == lidar_type)
        lidar_sensor_type = ORB_SLAM2::HDL_32;
    else if ("HDL_64" == lidar_type)
        lidar_sensor_type = ORB_SLAM2::HDL_64;
    else if ("HDL_64E" == lidar_type)
        lidar_sensor_type = ORB_SLAM2::HDL_64_EQUAL;
    else if ("RFANS_16" == lidar_type)
        lidar_sensor_type = ORB_SLAM2::RFANS_16;
    else if ("CFANS_32" == lidar_type)
        lidar_sensor_type = ORB_SLAM2::CFANS_32;

    // 地面提取
    /***************         地面提取PatchWork参数设置               *******************/
    ORB_SLAM2::_patchwork_param_t patchwork_param;
    patchwork_param.sensor_height_ = fSettings["patchwork.GPF.sensor_height"];
    fSettings["patchwork.verbose"] >> patchwork_param.verbose_;

    patchwork_param.num_iter_ = fSettings["patchwork.GPF.num_iter"];
    patchwork_param.num_lpr_ = fSettings["patchwork.GPF.num_lpr"];
    patchwork_param.num_min_pts_ = fSettings["patchwork.GPF.num_min_pts"];
    patchwork_param.th_seeds_ = fSettings["patchwork.GPF.th_seeds"];
    patchwork_param.th_dist_ = fSettings["patchwork.GPF.th_dist"];
    patchwork_param.max_range_ = fSettings["patchwork.GPF.max_r"];
    patchwork_param.min_range_ = fSettings["patchwork.GPF.min_r"];
    patchwork_param.num_rings_ = fSettings["patchwork.uniform.num_rings"];
    patchwork_param.num_sectors_ = fSettings["patchwork.uniform.num_sectors"];
    patchwork_param.uprightness_thr_ = fSettings["patchwork.GPF.uprightness_thr"];
    patchwork_param.adaptive_seed_selection_margin_ = fSettings["patchwork.adaptive_seed_selection_margin"];

    // For global threshold
    fSettings["patchwork.using_global_elevation"] >> patchwork_param.using_global_thr_;
    patchwork_param.global_elevation_thr_ = fSettings["patchwork.global_elevation_threshold"];

    patchwork_param.num_zones_ = fSettings["patchwork.czm.num_zones"];
    std::cout << "patchwork_param.num_zones_: " << patchwork_param.num_zones_ << std::endl;

    //num_sectors_each_zone
    cv::FileNode czm_num_sectors = fSettings["patchwork.czm.num_sectors_each_zone"];
    if (czm_num_sectors.type() != cv::FileNode::SEQ)
    {
        std::cerr << "num_sectors_each_zone is not a sequence" << std::endl;
    }
    cv::FileNodeIterator it_sector = czm_num_sectors.begin(), it_sector_end = czm_num_sectors.end();
    for (; it_sector != it_sector_end; it_sector++)
    {
        patchwork_param.num_sectors_each_zone_.push_back(*it_sector);
    }

    //num_rings_each_zone
    cv::FileNode czm_num_rings = fSettings["patchwork.czm.num_rings_each_zone"];
    if (czm_num_rings.type() != cv::FileNode::SEQ)
    {
        std::cerr << "num_rings_each_zone is not a sequence" << std::endl;
    }
    cv::FileNodeIterator it_ring = czm_num_rings.begin(), it_ring_end = czm_num_rings.end();
    for (; it_ring != it_ring_end; it_ring++)
    {
        patchwork_param.num_rings_each_zone_.push_back(*it_ring);
    }

    //min_ranges_
    cv::FileNode min_ranges = fSettings["patchwork.czm.min_ranges_each_zone"];
    if (min_ranges.type() != cv::FileNode::SEQ)
    {
        std::cerr << "min_ranges_each_zone is not a sequence" << std::endl;
    }
    cv::FileNodeIterator it_min_range = min_ranges.begin(), it_min_range_end = min_ranges.end();
    for (; it_min_range != it_min_range_end; it_min_range++)
    {
        patchwork_param.min_ranges_.push_back(*it_min_range);
    }

    // elevation_thr_
    cv::FileNode elevation_thresholds = fSettings["patchwork.czm.elevation_thresholds"];
    if (elevation_thresholds.type() != cv::FileNode::SEQ)
    {
        std::cerr << "elevation_thresholds is not a sequence" << std::endl;
    }
    cv::FileNodeIterator it_elevation_threshold = elevation_thresholds.begin(), it_elevation_threshold_end = elevation_thresholds.end();
    for (; it_elevation_threshold != it_elevation_threshold_end; it_elevation_threshold++)
    {
        patchwork_param.elevation_thr_.push_back(*it_elevation_threshold);
    }

    //flatness_thr_
    cv::FileNode flatness_thresholds = fSettings["patchwork.czm.flatness_thresholds"];
    if (flatness_thresholds.type() != cv::FileNode::SEQ)
    {
        std::cerr << "flatness_thresholds is not a sequence" << std::endl;
    }
    cv::FileNodeIterator it_flatness_threshold = flatness_thresholds.begin(), it_flatness_threshold_end = flatness_thresholds.end();
    for (; it_flatness_threshold != it_flatness_threshold_end; it_flatness_threshold++)
    {
        patchwork_param.flatness_thr_.push_back(*it_flatness_threshold);
    }

    ORB_SLAM2::LidarDepthExtration *mpLidarDepthExtractor = new ORB_SLAM2::LidarDepthExtration(depth_init_config, patchwork_param, Tcam_ldiar, depth_init_type, lidar_sensor_type,
                                                                                               fx, fy, cx, cy, width, height);
    ORB_SLAM2::_tracker_t tracker_;
    tracker_.levels = fSettings["Tracker.levels"];
    tracker_.min_level = fSettings["Tracker.min_level"];
    tracker_.max_level = fSettings["Tracker.max_level"];
    tracker_.max_iteration = fSettings["Tracker.max_iteration"];
    tracker_.scale_estimator = string(fSettings["Tracker.scale_estimator"]);
    tracker_.weight_function = string(fSettings["Tracker.weight_function"]);
    tracker_.set_scale_estimator_type();
    tracker_.set_weight_function_type();

    ORB_SLAM2::_reproj_tracker_t reproj_tracker_;
    reproj_tracker_.levels = fSettings["ReprojTracker.levels"];
    reproj_tracker_.max_iteration = fSettings["ReprojTracker.max_iteration"];
    reproj_tracker_.scale_estimator = string(fSettings["ReprojTracker.scale_estimator"]);
    reproj_tracker_.weight_function = string(fSettings["ReprojTracker.weight_function"]);
    reproj_tracker_.set_scale_estimator_type();
    reproj_tracker_.set_weight_function_type();

    // 直接法跟踪
    ORB_SLAM2::SparseLidarAlign *mpSparseLidarAlign = new ORB_SLAM2::SparseLidarAlign(mpPinholeCamera, tracker_);
    ORB_SLAM2::HybridSparseAlign *mpHybridSparseAlign = new ORB_SLAM2::HybridSparseAlign(mpPinholeCamera, tracker_, reproj_tracker_);
    ORB_SLAM2::SparseReprojAlign *mpSparseReprojAlign = new ORB_SLAM2::SparseReprojAlign(mpPinholeCamera, reproj_tracker_);

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesLidar;
    vector<double> vTimestamps;
    LoadImages(data_sequence, vstrImageFilenamesRGB, vstrImageFilenamesLidar, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        cerr << endl
             << "No images found in provided path." << endl;
        return 1;
    }
    else if (vstrImageFilenamesLidar.size() != vstrImageFilenamesRGB.size())
    {
        cerr << endl
             << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOLIDAR, true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    std::vector<Eigen::Matrix4d> vOdomPoses;

    cv::Mat mImGray;
    ORB_SLAM2::Frame lastFrame, currFrame;
    mImGray = cv::imread(vstrImageFilenamesRGB[0], CV_LOAD_IMAGE_GRAYSCALE);

    lastFrame = ORB_SLAM2::Frame(mImGray, vstrImageFilenamesLidar[0], vTimestamps[0], mpLidarDepthExtractor, mpORBextractorLeft, mpVocabulary,
                                 mpPinholeCamera, K, DistCoef, mbf, mThDepth);
    lastFrame.ComputeBoW();
    Sophus::SE3 Tcw(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    Sophus::SE3 Tcl(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    vOdomPoses.push_back(Eigen::Matrix4d::Identity());

    // Main loop
    int ni = 1;
    ros::Rate rate(100);
    while (ros::ok())
    {
        if (ni >= nImages)
        {
            static bool flag = false;
            if(!flag)
            {
                ROS_INFO("*******     completetd!      *********");
                flag = true;
            }
            ni++;
            rate.sleep();
            ros::spinOnce();
            continue;
        }

        // Read image and depthmap from file
        mImGray = cv::imread(vstrImageFilenamesRGB[ni], CV_LOAD_IMAGE_UNCHANGED);

        sensor_msgs::ImagePtr feature_img;
        feature_img = cv_bridge::CvImage(std_msgs::Header(), "mono8", mImGray).toImageMsg();
        pub_frame_img.publish(feature_img);

        double tframe = vTimestamps[ni];

        if (mImGray.empty())
        {
            cerr << endl
                 << "Failed to load image at: "
                 << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        currFrame = ORB_SLAM2::Frame(mImGray, vstrImageFilenamesLidar[ni], vTimestamps[ni], mpLidarDepthExtractor, mpORBextractorLeft, mpVocabulary,
                                     mpPinholeCamera, K, DistCoef, mbf, mThDepth);
        currFrame.ComputeBoW();

        // 特征匹配
        ORB_SLAM2::ORBmatcher orb_matcher(0.7, true);
        std::vector<int> vOrbMatches;
        int nmatches = orb_matcher.SearchByBoW(&lastFrame, currFrame, vOrbMatches);
        // ROS_INFO("feature match size: %d", nmatches);
        // DrawFeatureMatching(lastFrame, currFrame, vOrbMatches);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        // 直接法跟踪
        // mpSparseLidarAlign->tracking(&lastFrame, &currFrame, Tcl);

        // 间接法跟踪
        // mpSparseReprojAlign->tracking(&lastFrame, &currFrame, vOrbMatches, Tcl);

        mpHybridSparseAlign->hybrid_tracking(&lastFrame, &currFrame, vOrbMatches, Tcl);

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        double t_track = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
        std::cout << "t_track: " << t_track << std::endl;

        lastFrame = currFrame;

        Tcw = Tcl * Tcw;

        PublishTF(Tcw.inverse());

        vOdomPoses.push_back(Tcw.inverse().matrix());

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t1).count();

        vTimesTrack[ni] = t_track;

        // Wait to load the next frame
        // double T = 0;
        // if (ni < nImages - 1)
        //     T = vTimestamps[ni + 1] - tframe;
        // else if (ni > 0)
        //     T = tframe - vTimestamps[ni - 1];

        // if (ttrack < T)
        //     usleep((T - ttrack) * 1e6);

        ni++;
        ros::spinOnce();
    }

    // Stop all threads
    cout << "shutdown ORB_SLAM" << endl
         << endl;
    // SLAM.Shutdown();

    // // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl
         << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // // Save camera trajectory
    // // SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SaveTrajectoryKITTI("CameraTrajectoryKitti.txt", vOdomPoses);
    // SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<string> &vstrLidarFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixLidar = strPathToSequence + "/velodyne/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);
    vstrLidarFilenames.resize(nTimes);

    for (int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
        vstrLidarFilenames[i] = strPrefixLidar + ss.str() + ".bin";
    }
}

void PublishTF(Sophus::SE3 Twc)
{
    // PublishTF(Twc);
    Eigen::Matrix4d camera2world = Eigen::Matrix4d::Identity();
    camera2world.block<3, 3>(0, 0) = R_transform;

    Eigen::Matrix4d twc_camera = Twc.matrix();
    Eigen::Matrix4d twc_world = camera2world * twc_camera;

    // Eigen::Matrix4f twc             = Twc.matrix();
    Eigen::Vector3d cam_translation = twc_world.block<3, 1>(0, 3);
    Eigen::Matrix3d cam_rotation = twc_world.block<3, 3>(0, 0);
    Eigen::Quaterniond q_wc = Eigen::Quaterniond(cam_rotation);

    Eigen::Vector3d baselink_translation = R_transform * twc_camera.block<3, 1>(0, 3);
    Eigen::Matrix3d rotation = twc_camera.block<3, 3>(0, 0).transpose();
    Eigen::Vector3d euler_angles = rotation.eulerAngles(1, 0, 2); //ZXY顺序,相机坐标系中

    static tf::TransformBroadcaster odometery_tf_publisher;
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = ros::Time::now();
    odom_trans.header.frame_id = "/odom";
    odom_trans.child_frame_id = "/base_link";
    odom_trans.transform.translation.x = baselink_translation[0];
    odom_trans.transform.translation.y = baselink_translation[1];
    odom_trans.transform.translation.z = baselink_translation[2];
    odom_trans.transform.rotation = tf::createQuaternionMsgFromRollPitchYaw(euler_angles[2], euler_angles[1], euler_angles[0]);
    odometery_tf_publisher.sendTransform(odom_trans);

    //发布相机的pose
    // static tf::TransformBroadcaster odometery_tf_publisher;
    geometry_msgs::TransformStamped cam_pred_trans;
    cam_pred_trans.header.stamp = ros::Time::now();
    cam_pred_trans.header.frame_id = "/odom";
    cam_pred_trans.child_frame_id = "/camera_link";
    cam_pred_trans.transform.translation.x = cam_translation[0];
    cam_pred_trans.transform.translation.y = cam_translation[1];
    cam_pred_trans.transform.translation.z = cam_translation[2];
    cam_pred_trans.transform.rotation.x = q_wc.x();
    cam_pred_trans.transform.rotation.y = q_wc.y();
    cam_pred_trans.transform.rotation.z = q_wc.z();
    cam_pred_trans.transform.rotation.w = q_wc.w();
    odometery_tf_publisher.sendTransform(cam_pred_trans);

    //发布相机的轨迹
    geometry_msgs::PoseStamped lidar_odometry_posestamped;
    lidar_odometry_posestamped.pose.position.x = cam_translation(0);
    lidar_odometry_posestamped.pose.position.y = cam_translation(1);
    lidar_odometry_posestamped.pose.position.z = cam_translation(2);
    lidar_odometry_posestamped.pose.orientation.x = q_wc.x();
    lidar_odometry_posestamped.pose.orientation.y = q_wc.y();
    lidar_odometry_posestamped.pose.orientation.z = q_wc.z();
    lidar_odometry_posestamped.pose.orientation.w = q_wc.w();
    lidar_odometry_posestamped.header.stamp = ros::Time::now();
    lidar_odometry_posestamped.header.frame_id = "odom";
    lidar_odom_path.header.frame_id = "odom";
    lidar_odom_path.header.stamp = ros::Time::now();
    lidar_odom_path.poses.push_back(lidar_odometry_posestamped);

    pub_odom_path.publish(lidar_odom_path);
}

void DrawFeatureMatching(const ORB_SLAM2::Frame &frame1, const ORB_SLAM2::Frame &frame2, std::vector<int> &matches)
{
    cv::Mat image1 = 255 * frame1.mvImgPyramid[0].clone();
    cv::Mat image2 = 255 * frame2.mvImgPyramid[0].clone();

    int nRows = image1.rows;
    int nCols = image1.cols;

    cv::Mat out_img;
    cv::vconcat(image1, image2, out_img);
    if (out_img.channels() == 1)
        cv::cvtColor(out_img, out_img, CV_GRAY2BGR);

    for (size_t i = 0; i < matches.size(); i++)
    {
        int idx = matches.at(i);
        if (idx < 0)
            continue;

        float d = frame1.mvDepth[i];

        cv::Point2f pt_1 = frame1.mvKeys[i].pt;
        cv::Point2f pt_2 = frame2.mvKeys[idx].pt;

        if (d < 0)
        {
            cv::circle(out_img, pt_1, 2, cv::Scalar(255, 0, 0), -1);
            cv::circle(out_img, pt_2 + cv::Point2f(0, nRows), 2, cv::Scalar(255, 0, 0), -1);
            cv::line(out_img, pt_1, pt_2 + cv::Point2f(0, nRows), cv::Scalar(0, 255, 0), 1, CV_AA);
        }
        else
        {
            cv::circle(out_img, pt_1, 2, cv::Scalar(255, 0, 0), -1);
            cv::circle(out_img, pt_2 + cv::Point2f(0, nRows), 2, cv::Scalar(255, 0, 0), -1);
            cv::line(out_img, pt_1, pt_2 + cv::Point2f(0, nRows), cv::Scalar(0, 0, 255), 1, CV_AA);
        }
    }
    cv::imwrite("/home/kinggreat24/pc/feature_match.png", out_img);
    // cv::imshow("feature_match", out_img);
    // cv::waitKey(1);
}

void SaveTrajectoryKITTI(const string &filename, const std::vector<Eigen::Matrix4d> &vOdomPoses)
{
    cout << endl
         << "Saving camera trajectory to " << filename << " ..." << endl;

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).

    for (int i = 0; i < vOdomPoses.size(); i++)
    {
        Eigen::Matrix4d twc = vOdomPoses[i];
        f << setprecision(9) << twc(0, 0) << " " << twc(0, 1) << " " << twc(0, 2) << " " << twc(0, 3) << " " << twc(1, 0) << " " << twc(1, 1) << " " << twc(1, 2) << " " << twc(1, 3) << " " << twc(2, 0) << " " << twc(2, 1) << " " << twc(2, 2) << " " << twc(2, 3) << endl;
    }
    f.close();
    cout << endl
         << "trajectory saved!" << endl;
}