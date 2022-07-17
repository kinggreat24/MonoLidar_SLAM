/*
 * @Author: kinggreat24
 * @Date: 2022-07-06 12:19:13
 * @LastEditTime: 2022-07-08 13:39:17
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /ORB_SLAM2/include/LidarDepthExtraction.h
 * 可以输入预定的版权声明、个性签名、空行等
 */
#ifndef LIDAR_DEPTH_EXTRACTION_H
#define LIDAR_DEPTH_EXTRACTION_H

#include "monolidar_fusion/DepthEstimator.h"
#include "monolidar_fusion/camera_pinhole.h"

#include "include_fade25d/Fade_2D.h"

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h> //条件滤波
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h> //体素滤波器头文件
#include <pcl/filters/crop_box.h>

#include <unordered_map>

#include "Common.h"
#include "patchwork.hpp"

#include "utils/velodyne_utils.h"
#include "ground_removal/depth_ground_remover.h"
#include "clusterers/image_based_clusterer.h"
#include "image_labelers/diff_helpers/diff_factory.h"

namespace ORB_SLAM2
{
    class LidarDepthExtration
    {
    public:
        LidarDepthExtration(const std::string &setting_file, const _patchwork_param_t &patchwork_param, const Eigen::Matrix4d &Tcam_lidar,
                            const eDepthInitMethod depth_inti_type, const eLidarSensorType lidar_sensor_type,
                            const double fx, const double fy, const double cx, const double cy, const double width, const double height);

        void CalculateFeatureDepthsCurFrame(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in_cur,
                                            const std::vector<cv::KeyPoint> &kps, Eigen::VectorXd &depths);

        void CalculateFeatureDepthsCamVox(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in_cur,
                                          const std::vector<cv::KeyPoint> &kps, std::vector<float> &vDepths);

        GEOM_FADE25D::Fade_2D *CreateTerrain(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_cam);

        GEOM_FADE25D::Fade_2D *CreateTerrain(std::unordered_map<uint16_t, depth_clustering::Cloud> &clusters);

        void DrawObjectDelauny(GEOM_FADE25D::Fade_2D *pdt_all, const std::vector<cv::KeyPoint> &mvKeys, std::string file_name);

        int PointFeatureDepthInit(GEOM_FADE25D::Fade_2D *pdt_all, const std::vector<cv::KeyPoint> &orb_features,
                                  std::vector<float> &vDepths);

        int PointFeatureDepthInit(GEOM_FADE25D::Fade_2D *pdt_obj, GEOM_FADE25D::Fade_2D *pdt_ground,
                                  const std::vector<cv::KeyPoint> &orb_features, std::vector<float> &vDepths);

        void pointCloudDepthFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr, const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                                   const char *field, const float minDepth, const float maxDepth);

        // 激光地面点云分割
        void PatchWorkGroundSegmentation(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_in, pcl::PointCloud<pcl::PointXYZI>::Ptr &ground_cloud,
            pcl::PointCloud<pcl::PointXYZI>::Ptr &obstacle_cloud, double time_takens);

        // 激光点云目标三角网构造
        void LidarDepthClustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pObstacleCloud,
                                  std::unordered_map<uint16_t, depth_clustering::Cloud> &vLidarClusters, cv::Mat &rangeSegImage);

    public:
        Eigen::Matrix4d Tcam_lidar_; //激光雷达与相机之间的外参
        const eDepthInitMethod depth_init_type_;
        const eLidarSensorType lidar_sensor_type_;
    protected:
        boost::shared_ptr<PatchWork<pcl::PointXYZI>> mpPatchworkGroundSeg;      //地面点云提取
        std::unique_ptr<depth_clustering::ProjectionParams> mpProjectionParams; //点云投影参数

    private:
        Mono_Lidar::DepthEstimatorParameters depth_estimator_parameters_;
        Mono_Lidar::GroundPlane::Ptr ground_plane_;
        Mono_Lidar::DepthEstimator depth_estimator_;
        std::shared_ptr<CameraPinhole> cam_pinhole_;

        cv::Size2i mImgSize_;
        cv::Mat mK_;
    };
} // namespace ORB_SLAM2

#endif