/*
 * @Author: your name
 * @Date: 2021-09-16 21:21:47
 * @LastEditTime: 2022-07-12 10:33:08
 * @LastEditors: kinggreat24
 * @Description: In User Settings Edit
 * @FilePath: /ORB_SLAM2/Examples/Monocular-Lidar/hybrid_track_ws/src/hybrid_vlo/src/hybrid_visual_odometry.cpp
 */

#include <ros/ros.h>

#include "utils.h"
#include "Frame.h"

#include "lidar_sparse_align/SparseLidarAlign.h"
#include "lidar_sparse_align/FeatureAlign.h"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <geometry_msgs/TransformStamped.h>

#include <opencv2/core/eigen.hpp>

// global variables
Eigen::Matrix4d cameraLidarExtrinsic = Eigen::Matrix4d::Identity();
vk::PinholeCamera *pPinhole_camera = static_cast<vk::PinholeCamera *>(NULL);
std::vector<Eigen::Matrix4d> global_gt_poses_;

ros::Publisher pub_image_, pub_disparity_, pub_path_, pub_odometry_;

nav_msgs::Path stereo_odom_path_;

//Camera 2 world coordinate
Eigen::Matrix3d R_transform; // camera_link 2 base_link
Eigen::Quaterniond q_transform;

std::vector<Eigen::Matrix4d> v_odom_poses_;


// global functions
void advertise_tf_odom(const Eigen::Matrix4d &Tcw);
void SaveTrajectoryKITTI(const string &filename);


int main(int argc, char **argv)
{
    ros::init(argc, argv, "hybrid_vlo");
    ros::NodeHandle nh, nh_private("~");

    pub_image_ = nh.advertise<sensor_msgs::Image>("image_left", 1);
    pub_disparity_ = nh.advertise<sensor_msgs::Image>("image_disparity", 1);
    pub_path_ = nh.advertise<nav_msgs::Path>("odom_path", 1);
    pub_odometry_ = nh.advertise<nav_msgs::Odometry>("odom", 1);

    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    q_transform = Eigen::Quaterniond(R_transform);

    std::string strTrajectorySavePath("");
    nh_private.param("strTrajectorySavePath", strTrajectorySavePath, strTrajectorySavePath);

    std::string strSettingPath("");
    nh_private.param("strSettingPath", strSettingPath, strSettingPath);
    if (strSettingPath.empty())
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
    std::vector<std::string> vstrImageDisparity;
    std::vector<std::string> vstrLidar;
    std::vector<double> vTimestamps;
    std::string data_sequene = (std::string)fSettings["data_sequence"];
    hybrid_vlo::LoadImages(data_sequene, vstrImageLeft, vstrImageRight, vstrImageDisparity, vstrLidar, vTimestamps);
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
    ORB_SLAM2::ORBextractor *pInitOrbExtractor = new ORB_SLAM2::ORBextractor(nFeatures, scale_factor, nlevels, iniThFAST, minThFAST);
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
    mpFeatureAlign->verbose_ = false;

    hybrid_vlo::Frame *pLastFrame, *pCurrentFrame;
    // 上一帧到当前帧的变换
    Sophus::SE3 Tcl = Sophus::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    //当前相机的姿态
    Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
    cv::Mat img_last_L, img_last_R, img_cur_L, img_cur_R;


    bool use_disparity = false;

    ros::Rate rate(500);
    const int start_frame = 0;
    int ni = start_frame;
    while (ros::ok())
    {
        if (ni < nImages)
        {
            //
            img_cur_L = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
            
            if(use_disparity)
                img_cur_R = cv::imread(vstrImageDisparity[ni], CV_LOAD_IMAGE_UNCHANGED);
            else
                img_cur_R = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);
            
            
            if (img_cur_L.empty() || img_cur_R.empty())
                return -1;

            if (ni == start_frame)
            {
                
                pCurrentFrame = new hybrid_vlo::Frame(img_cur_L, img_cur_R, vstrLidar[ni],
                    pInitOrbExtractor, pPinhole_camera, nLevels, cameraLidarExtrinsic, bf, use_disparity);

                // 特征激光点云深度拟合
                std::chrono::steady_clock::time_point t_feature_depth1 = std::chrono::steady_clock::now();
                GEOM_FADE25D::Fade_2D * pdt = pCurrentFrame->CreateTerrain(pCurrentFrame->mpLidarPointCloudCamera);
                pCurrentFrame->mvDepths.resize(pCurrentFrame->mvKeys.size(),-1);
                for(size_t i=0;i<pCurrentFrame->mvKeys.size();i++)
                {
                    float depth = pCurrentFrame->DepthFitting(pdt,pCurrentFrame->mvKeys.at(i).pt);
                    pCurrentFrame->mvDepths[i] = depth;
                }
                std::chrono::steady_clock::time_point t_feature_depth2 = std::chrono::steady_clock::now();
                double tfeature_depth= std::chrono::duration_cast<std::chrono::duration<double> >(t_feature_depth2 - t_feature_depth1).count();
                std::cout<<"feature depth extractuion: "<<tfeature_depth<<std::endl;


                img_last_L = img_cur_L;
                img_last_R = img_cur_R;
                pLastFrame = pCurrentFrame;

                advertise_tf_odom(Tcw);

                ros::spinOnce();
                rate.sleep();

                ni++;
                continue;
            }

            // tracking
            pCurrentFrame = new hybrid_vlo::Frame(img_cur_L, img_cur_R, vstrLidar[ni],
                                                  pOrbExtractor, pPinhole_camera, nLevels, cameraLidarExtrinsic, bf, use_disparity);
            
            // 特征激光点云深度拟合
            std::chrono::steady_clock::time_point t_feature_depth1 = std::chrono::steady_clock::now();
            GEOM_FADE25D::Fade_2D * pdt = pCurrentFrame->CreateTerrain(pCurrentFrame->mpLidarPointCloudCamera);
            pCurrentFrame->mvDepths.resize(pCurrentFrame->mvKeys.size(),-1);
            for(size_t i=0;i<pCurrentFrame->mvKeys.size();i++)
            {
                float depth = pCurrentFrame->DepthFitting(pdt,pCurrentFrame->mvKeys.at(i).pt);
                pCurrentFrame->mvDepths[i] = depth;
            }
            std::chrono::steady_clock::time_point t_feature_depth2 = std::chrono::steady_clock::now();
            double tfeature_depth= std::chrono::duration_cast<std::chrono::duration<double> >(t_feature_depth2 - t_feature_depth1).count();
            std::cout<<"feature depth extractuion: "<<tfeature_depth<<std::endl;


            // 特征跟踪
            std::vector<cv::Point2f> prepoints, nextpoints;
            std::vector<uchar> track_state;
            cv::Mat track_results;
            hybrid_vlo::klt_feature_tracking(img_last_L, img_cur_L, pLastFrame->mvKeys, prepoints, nextpoints, track_state, cv::Mat(), track_results, true);

            // 显示特征跟踪的结果
            track_results = hybrid_vlo::showTrackingResults(img_last_L, img_cur_L, prepoints, nextpoints, track_state);
            std::cout << "last frame id: " << pLastFrame->mnId << std::endl;
            sensor_msgs::ImagePtr feature_img;
            feature_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", track_results).toImageMsg();
            pub_image_.publish(feature_img);

            
            // 基于特征的运动估计
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            mpFeatureAlign->tracking(prepoints, pLastFrame->mvDepths, nextpoints, track_state, Tcl);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
            std::cout << "feature tracking time: " << ttrack << std::endl;

            // 视觉激光融合直接法跟踪
            // std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
            // mpLidarSparseAlign->tracking(pLastFrame, pCurrentFrame, Tcl);
            // std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
            // double tlidartrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
            // std::cout<<"direct tracking time: "<<tlidartrack<<std::endl;


            // 更新位置信息
            Tcw = Tcl.matrix() * Tcw;


            img_last_L = img_cur_L;
            img_last_R = img_cur_R;
            
            delete pLastFrame;
            pLastFrame = pCurrentFrame;

            advertise_tf_odom(Tcw);

            ros::spinOnce();
            rate.sleep();
            ni++;
        }
        else
        {
            static bool is_trajectory_saved = false;
            if(!is_trajectory_saved)
            {
                is_trajectory_saved = true;
                SaveTrajectoryKITTI(strTrajectorySavePath);
            }
            ros::spinOnce();
            rate.sleep();
        }
    }

    return 0;
}

void advertise_tf_odom(const Eigen::Matrix4d &Tcw)
{
    Eigen::Matrix4d twc = Tcw.inverse();
    v_odom_poses_.push_back(twc);

    Eigen::Vector3d translation = R_transform * twc.block<3, 1>(0, 3);
    Eigen::Matrix3d rotation = twc.block<3, 3>(0, 0).transpose();

    Eigen::Vector3d euler_angles = rotation.eulerAngles(1, 0, 2); //ZXY顺序,相机坐标系中

    static tf::TransformBroadcaster odometery_tf_publisher;
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = ros::Time::now();
    odom_trans.header.frame_id = "/odom";
    odom_trans.child_frame_id = "/base_link";
    odom_trans.transform.translation.x = translation[0];
    odom_trans.transform.translation.y = translation[1];
    odom_trans.transform.translation.z = translation[2];
    odom_trans.transform.rotation = tf::createQuaternionMsgFromRollPitchYaw(euler_angles[2], euler_angles[1], euler_angles[0]);
    odometery_tf_publisher.sendTransform(odom_trans);

    geometry_msgs::PoseStamped stereo_odometry_posestamped;
    stereo_odometry_posestamped.pose.position.x = translation(0);
    stereo_odometry_posestamped.pose.position.y = translation(1);
    stereo_odometry_posestamped.pose.position.z = translation(2);

    Eigen::Quaterniond q_w_i(twc.topLeftCorner<3, 3>());
    Eigen::Quaterniond q = q_transform * q_w_i;
    q.normalize();
    stereo_odometry_posestamped.pose.orientation.x = q.x();
    stereo_odometry_posestamped.pose.orientation.y = q.y();
    stereo_odometry_posestamped.pose.orientation.z = q.z();
    stereo_odometry_posestamped.pose.orientation.w = q.w();

    stereo_odometry_posestamped.header.stamp = ros::Time::now();
    stereo_odometry_posestamped.header.frame_id = "odom";

    stereo_odom_path_.header.frame_id = "odom";
    stereo_odom_path_.header.stamp = ros::Time::now();
    stereo_odom_path_.poses.push_back(stereo_odometry_posestamped);

    pub_path_.publish(stereo_odom_path_);

    std::cout << "Twc: " << std::endl
              << twc.matrix() << std::endl;
}


void SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
   
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
   
    for(int i=0;i<v_odom_poses_.size();i++)
    {
        Eigen::Matrix4d twc = v_odom_poses_[i];
        f << setprecision(9) << 
            twc(0,0) << " " << twc(0,1)  << " " << twc(0,2) << " "  << twc(0,3) << " " <<
            twc(1,0) << " " << twc(1,1)  << " " << twc(1,2) << " "  << twc(1,3) << " " <<
            twc(2,0) << " " << twc(2,1)  << " " << twc(2,2) << " "  << twc(2,3) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}