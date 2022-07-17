/*
 * @Author: kinggreat24
 * @Date: 2022-07-06 12:28:20
 * @LastEditTime: 2022-07-09 10:15:44
 * @LastEditors: kinggreat24
 * @Description: 激光点云拟合视觉特征的深度信息
 * @FilePath: /ORB_SLAM2/src/LidarDepthExtraction.cc
 * 可以输入预定的版权声明、个性签名、空行等
 */
#include "LidarDepthExtraction.h"
#include "Converter.h"

namespace ORB_SLAM2
{
    LidarDepthExtration::LidarDepthExtration(const std::string &_path_config_depthEstimator, const _patchwork_param_t &patchwork_param,
                                             const Eigen::Matrix4d &Tcam_lidar, const eDepthInitMethod depth_inti_type, const eLidarSensorType lidar_sensor_type,
                                             const double fx, const double fy, const double cx, const double cy, const double width, const double height)
        : depth_init_type_(depth_inti_type), lidar_sensor_type_(lidar_sensor_type)
    {
        std::cout << "Read setting file: " << _path_config_depthEstimator << std::endl;

        depth_estimator_.InitConfig(_path_config_depthEstimator);

        double focal_length = (fx + fy) / 2;
        double principal_point_x = cx;
        double principal_point_y = cy;
        std::cout << "Camera parameters: " << std::endl
                  << " Camera.fx: " << fx << "  Camera.fy: " << fx << " Camera.cx: " << principal_point_x << " Camera.cy: " << principal_point_y << std::endl;
        cam_pinhole_.reset(new CameraPinhole(width, height, focal_length, principal_point_x, principal_point_y));

        Eigen::Affine3d T_cam_lidar(Tcam_lidar);
        depth_estimator_.Initialize(cam_pinhole_, T_cam_lidar);

        depth_estimator_parameters_.fromFile(_path_config_depthEstimator);
        double plane_inlier_threshold = depth_estimator_parameters_.ransac_plane_refinement_treshold;

        // 地面提取
        mpPatchworkGroundSeg.reset(new PatchWork<pcl::PointXYZI>(patchwork_param));

        // 激光点云图像投影参数
        if (VLP_16 == lidar_sensor_type_)
            mpProjectionParams = depth_clustering::ProjectionParams::VLP_16();
        else if (HDL_32 == lidar_sensor_type_)
            mpProjectionParams = depth_clustering::ProjectionParams::HDL_32();
        else if (HDL_64 == lidar_sensor_type_)
            mpProjectionParams = depth_clustering::ProjectionParams::HDL_64();
        else if (HDL_64_EQUAL == lidar_sensor_type_)
            mpProjectionParams = depth_clustering::ProjectionParams::HDL_64_EQUAL();
        else if (RFANS_16 == lidar_sensor_type_)
            mpProjectionParams = depth_clustering::ProjectionParams::RFANS_16();
        else if (CFANS_32 == lidar_sensor_type_)
            mpProjectionParams = depth_clustering::ProjectionParams::CFANS_32();

        // 图像相关参数
        mImgSize_ = cv::Size2i(width, height);

        mK_ = cv::Mat::zeros(3, 3, CV_32F);
        mK_.at<float>(0, 0) = fx;
        mK_.at<float>(1, 1) = fy;
        mK_.at<float>(0, 2) = cx;
        mK_.at<float>(1, 2) = cy;

        Tcam_lidar_ = Tcam_lidar;
    }

    // 使用LIMO深度拟合
    void LidarDepthExtration::CalculateFeatureDepthsCurFrame(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in_cur,
                                                             const std::vector<cv::KeyPoint> &kps, Eigen::VectorXd &depths)
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

        // Init plane ransac
        ground_plane_ = std::make_shared<Mono_Lidar::RansacPlane>(
            std::make_shared<Mono_Lidar::DepthEstimatorParameters>(depth_estimator_parameters_));
        depth_estimator_.CalculateDepth(cloud_in_cur, featureCoordinates, depths, ground_plane_);
    }

    void LidarDepthExtration::CalculateFeatureDepthsCamVox(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in_cur,
                                                           const std::vector<cv::KeyPoint> &kps, std::vector<float> &vDepths)
    {
        cv::Mat imDepth = cv::Mat::zeros(mImgSize_, CV_32F);

        float fx = mK_.at<float>(0, 0);
        float fy = mK_.at<float>(1, 1);
        float cx = mK_.at<float>(0, 2);
        float cy = mK_.at<float>(1, 2);

        for (int i = 0; i < cloud_in_cur->size(); i++)
        {
            pcl::PointXYZI pt = cloud_in_cur->at(i);
            if (pt.z < 0)
                continue;

            float u = fx * pt.x / pt.z + cx;
            float v = fy * pt.y / pt.z + cy;

            if (u < 0 || u >= imDepth.cols || v < 0 || v >= imDepth.rows)
                continue;

            imDepth.at<float>(v, u) = pt.z;
        }

        for (size_t i = 0; i < kps.size(); i++)
        {
            cv::KeyPoint kpt = kps.at(i);

            float d = imDepth.at<float>(kpt.pt.y, kpt.pt.x);

            if (d > 0)
            {
                vDepths[i] = d;
            }
        }
    }

    void LidarDepthExtration::pointCloudDepthFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_Condition,
                                                    const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, const char *field,
                                                    const float minDepth, const float maxDepth)
    {
        pcl::PassThrough<pcl::PointXYZI> pass_through;
        pass_through.setInputCloud(cloud);
        pass_through.setFilterFieldName("z");
        pass_through.setFilterLimits(minDepth, maxDepth); //-0.25<x<0.15 为内点
        pass_through.filter(*cloud_after_Condition);
    }

    int LidarDepthExtration::PointFeatureDepthInit(GEOM_FADE25D::Fade_2D *pdt_all, const std::vector<cv::KeyPoint> &orb_features,
                                                   std::vector<float> &vDepths)
    {
        int valid_depth = 0;
        // #pragma omp parallel for
        for (size_t i = 0; i < orb_features.size(); i++)
        {
            cv::Point2f pt = orb_features[i].pt;
            GEOM_FADE25D::Point2 pfeature(pt.x, mImgSize_.height - pt.y, 0);
            GEOM_FADE25D::Triangle2 *pTriangle = pdt_all->locate(pfeature);

            Eigen::Vector3d normalVector;
            Eigen::Vector3d AB, AC, BC;

            GEOM_FADE25D::Point2 *pA, *pB, *pC;
            //落在物体构成的三角网上
            if (pTriangle)
            {
                //只用当前三角网进行深度拟合
                pA = pTriangle->getCorner(0);
                pB = pTriangle->getCorner(1);
                pC = pTriangle->getCorner(2);

                AB = Eigen::Vector3d(pB->x() - pA->x(), pB->y() - pA->y(), pB->z() - pA->z());
                AC = Eigen::Vector3d(pC->x() - pA->x(), pC->y() - pA->y(), pC->z() - pA->z());
                BC = Eigen::Vector3d(pC->x() - pB->x(), pC->y() - pB->y(), pC->z() - pB->z());

                //三角网长度不能太长
                if (AB.x() > 30 || AB.y() > 30 || AC.x() > 30 || AC.y() > 30 || BC.x() > 30 || BC.y() > 30)
                    continue;
            }
            else
                continue;

            normalVector = AB.cross(AC);
            normalVector.normalize();

            Eigen::Vector3d AP(pfeature.x() - pA->x(), pfeature.y() - pA->y(), pfeature.z() - pA->z());
            float depth = -(normalVector(0) * AP(0) + normalVector(1) * AP(1)) / normalVector(2) + pA->z();
            vDepths[i] = depth;
            valid_depth++;
        }
        return valid_depth;
    }

    int LidarDepthExtration::PointFeatureDepthInit(GEOM_FADE25D::Fade_2D *pdt_obj, GEOM_FADE25D::Fade_2D *pdt_ground,
                                                   const std::vector<cv::KeyPoint> &orb_features, std::vector<float> &vDepths)
    {
        int valid_depth = 0;

        for (size_t i = 0; i < orb_features.size(); i++)
        {
            cv::Point2f pt = orb_features[i].pt;
            GEOM_FADE25D::Point2 pfeature(pt.x, mImgSize_.height - pt.y, 0);
            GEOM_FADE25D::Triangle2 *pTriangle = pdt_obj->locate(pfeature);

            Eigen::Vector3d normalVector;
            Eigen::Vector3d AB, AC, BC;

            GEOM_FADE25D::Point2 *pA, *pB, *pC;
            //落在物体构成的三角网上
            if (pTriangle)
            {
                //只用当前三角网进行深度拟合
                pA = pTriangle->getCorner(0);
                pB = pTriangle->getCorner(1);
                pC = pTriangle->getCorner(2);

                //如果特征点不在同一个物体上
                if (pA->getCustomIndex() != pB->getCustomIndex() || pA->getCustomIndex() != pC->getCustomIndex() || pB->getCustomIndex() != pC->getCustomIndex())
                {
                    //判断是否属于地面
                    GEOM_FADE25D::Triangle2 *pGroundTriangle = pdt_ground->locate(pfeature);
                    if (pGroundTriangle)
                    {
                        //落在地面上
                        pA = pGroundTriangle->getCorner(0);
                        pB = pGroundTriangle->getCorner(1);
                        pC = pGroundTriangle->getCorner(2);

                        //如果特征点不在同一个物体上
                        if (pA->getCustomIndex() != pB->getCustomIndex() || pA->getCustomIndex() != pC->getCustomIndex() || pB->getCustomIndex() != pC->getCustomIndex())
                            continue;

                        AB = Eigen::Vector3d(pB->x() - pA->x(), pB->y() - pA->y(), pB->z() - pA->z());
                        AC = Eigen::Vector3d(pC->x() - pA->x(), pC->y() - pA->y(), pC->z() - pA->z());
                        BC = Eigen::Vector3d(pC->x() - pB->x(), pC->y() - pB->y(), pC->z() - pB->z());

                        //三角网长度不能太长
                        if (AB.x() > 30 || AB.y() > 30 || AC.x() > 30 || AC.y() > 30 || BC.x() > 30 || BC.y() > 30)
                            continue;
                    }
                }

                AB = Eigen::Vector3d(pB->x() - pA->x(), pB->y() - pA->y(), pB->z() - pA->z());
                AC = Eigen::Vector3d(pC->x() - pA->x(), pC->y() - pA->y(), pC->z() - pA->z());
                BC = Eigen::Vector3d(pC->x() - pB->x(), pC->y() - pB->y(), pC->z() - pB->z());

                //深度变化不能太大
                if (fabs(AB.z()) > 5 || fabs(AC.z()) > 5 || fabs(BC.z()) > 5)
                    continue;
            }
            else
                continue;

            normalVector = AB.cross(AC);
            normalVector.normalize();

            Eigen::Vector3d AP(pfeature.x() - pA->x(), pfeature.y() - pA->y(), pfeature.z() - pA->z());
            float depth = -(normalVector(0) * AP(0) + normalVector(1) * AP(1)) / normalVector(2) + pA->z();
            vDepths[i] = depth;
            valid_depth++;
        }
        return valid_depth;
    }

    // 将所有的点生成
    GEOM_FADE25D::Fade_2D *LidarDepthExtration::CreateTerrain(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_cam)
    {
        GEOM_FADE25D::Fade_2D *pDt = new GEOM_FADE25D::Fade_2D();
        float fx = mK_.at<float>(0, 0);
        float fy = mK_.at<float>(1, 1);
        float cx = mK_.at<float>(0, 2);
        float cy = mK_.at<float>(1, 2);

        //将点云转到相机坐标空间
        int NP = (int)pc_cam->size();
        // std::cout << "NP： " << NP << std::endl;
        std::vector<GEOM_FADE25D::Point2> vPoints;
        vPoints.reserve(NP);

        // #pragma omp parallel for
        for (int i = 0; i < NP; i++)
        {
            pcl::PointXYZI pt = pc_cam->points.at(i);
            if (pt.z < 0)
                continue;

            //投影到图像上
            float u = fx * pt.x / pt.z + cx;
            float v = fy * pt.y / pt.z + cy;

            if (u >= mImgSize_.width || u < 0 || v >= mImgSize_.height || v < 0)
                continue;

            GEOM_FADE25D::Point2 p(u, mImgSize_.height - v, pt.z);
            p.setCustomIndex(0);

            vPoints.push_back(p);
        }

        pDt->insert(vPoints);
        return pDt;
    }

    GEOM_FADE25D::Fade_2D *LidarDepthExtration::CreateTerrain(std::unordered_map<uint16_t, depth_clustering::Cloud> &clusters)
    {
        GEOM_FADE25D::Fade_2D *pDt = new GEOM_FADE25D::Fade_2D();
        float fx = mK_.at<float>(0, 0);
        float fy = mK_.at<float>(1, 1);
        float cx = mK_.at<float>(0, 2);
        float cy = mK_.at<float>(1, 2);

        std::vector<GEOM_FADE25D::Point2> vPoints;
        int cluster_idx = 0;
        for (auto cluster_iter = clusters.begin(); cluster_iter != clusters.end(); cluster_iter++)
        {
            //
            pcl::PointCloud<pcl::PointXYZI>::Ptr pc_cluster = cluster_iter->second.ToPcl(cluster_iter->first);

            //将点云转到相机坐标空间
            pcl::PointCloud<pcl::PointXYZI>::Ptr pc_cluster_cam(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::transformPointCloud(*pc_cluster, *pc_cluster_cam, Tcam_lidar_);

            for (int i = 0; i < pc_cluster_cam->size(); i++)
            {
                pcl::PointXYZI pt = pc_cluster_cam->points.at(i);
                if (pt.z < 0)
                    continue;

                //投影到图像上
                float u = fx * pt.x / pt.z + cx;
                float v = fy * pt.y / pt.z + cy;

                if (u >= mImgSize_.width || u < 0 || v >= mImgSize_.height || v < 0)
                    continue;

                GEOM_FADE25D::Point2 p(u, mImgSize_.height - v, pt.z);
                p.setCustomIndex(cluster_idx);
                vPoints.push_back(p);
            }
            cluster_idx++;
        }
        // GEOM_FADE25D::EfficientModel em(vPoints);
        // vPoints.clear();
        // double maxError(.1);
        // em.extract(maxError,vPoints);
        pDt->insert(vPoints);
        return pDt;
    }

    void LidarDepthExtration::DrawObjectDelauny(GEOM_FADE25D::Fade_2D *pdt_all, const std::vector<cv::KeyPoint> &mvKeys, std::string file_name)
    {
        GEOM_FADE25D::Visualizer2 vis(file_name);

        int height = mImgSize_.height;
        int width = mImgSize_.width;

        // Some colors.
        GEOM_FADE25D::Color cBlack(GEOM_FADE25D::CBLACK);
        GEOM_FADE25D::Color cBlue(GEOM_FADE25D::CBLUE);
        GEOM_FADE25D::Color cGreen(GEOM_FADE25D::CGREEN);
        GEOM_FADE25D::Color cRed(GEOM_FADE25D::CRED);
        GEOM_FADE25D::Color cYellow(GEOM_FADE25D::CYELLOW);

        //随机颜色
        std::vector<GEOM_FADE25D::Color> randomColors(100);
        cv::RNG rng;
        for (int i = 0; i < randomColors.size(); i++)
        {
            GEOM_FADE25D::Color random_color(rng.uniform(0, 255) / 255.0, rng.uniform(0, 255) / 255.0, rng.uniform(0, 255) / 255.0, 0.001);
            randomColors[i] = random_color;
        }

        //绘制图像边界
        GEOM_FADE25D::Point2 p_top_left(0, height, 0);
        GEOM_FADE25D::Point2 p_top_right(width, height, 0);
        GEOM_FADE25D::Point2 p_buttom_left(0, 0, 0);
        GEOM_FADE25D::Point2 p_buttom_right(width, 0, 0);
        vis.addObject(GEOM_FADE25D::Segment2(p_top_left, p_top_right), cRed);
        vis.addObject(GEOM_FADE25D::Segment2(p_top_left, p_buttom_left), cRed);
        vis.addObject(GEOM_FADE25D::Segment2(p_buttom_left, p_buttom_right), cRed);
        vis.addObject(GEOM_FADE25D::Segment2(p_top_right, p_buttom_right), cRed);

        //get all triangles
        std::vector<GEOM_FADE25D::Triangle2 *> vAllDelaunayTriangles;
        pdt_all->getTrianglePointers(vAllDelaunayTriangles);
        //同一个物体内部的颜色一致
        for (std::vector<GEOM_FADE25D::Triangle2 *>::iterator it = vAllDelaunayTriangles.begin(); it != vAllDelaunayTriangles.end(); ++it)
        {
            GEOM_FADE25D::Triangle2 *pT(*it);

            // An alternative method (just to show how to access the vertices) would be:
            GEOM_FADE25D::Point2 *p0 = pT->getCorner(0);
            GEOM_FADE25D::Point2 *p1 = pT->getCorner(1);
            GEOM_FADE25D::Point2 *p2 = pT->getCorner(2);

            if (p0->getCustomIndex() == p1->getCustomIndex())
                vis.addObject(GEOM_FADE25D::Segment2(*p0, *p1), randomColors[p0->getCustomIndex()]);
            // else
            //     vis.addObject(GEOM_FADE25D::Segment2(*p0, *p1), cBlack);

            if (p0->getCustomIndex() == p2->getCustomIndex())
                vis.addObject(GEOM_FADE25D::Segment2(*p0, *p2), randomColors[p0->getCustomIndex()]);
            // else
            //     vis.addObject(GEOM_FADE25D::Segment2(*p0, *p2), cBlack);

            if (p1->getCustomIndex() == p2->getCustomIndex())
                vis.addObject(GEOM_FADE25D::Segment2(*p1, *p2), randomColors[p1->getCustomIndex()]);
            // else
            //     vis.addObject(GEOM_FADE25D::Segment2(*p1, *p2), cBlack);
        }

        //绘制目标检测的结果
        // std::cout<<"draw object bounding box"<<std::endl;
        // for(auto obj_iter = obj_box_lists.begin(); obj_iter != obj_box_lists.end(); obj_iter++)
        // {
        //     //绘制bounding box
        //     GEOM_FADE25D::Point2 bbox_top_left(obj_iter->x, height - obj_iter->y, 0);
        //     GEOM_FADE25D::Point2 bbox_top_right(obj_iter->x + obj_iter->w, height - obj_iter->y , 0);
        //     GEOM_FADE25D::Point2 bbox_buttom_left(obj_iter->x, height - obj_iter->y - obj_iter->h, 0);
        //     GEOM_FADE25D::Point2 bbox_buttom_right(obj_iter->x + obj_iter->w, height - obj_iter->y - obj_iter->h, 0);
        //     vis.addObject(GEOM_FADE25D::Segment2(bbox_top_left, bbox_top_right), cRed);
        //     vis.addObject(GEOM_FADE25D::Segment2(bbox_top_left, bbox_buttom_left), cRed);
        //     vis.addObject(GEOM_FADE25D::Segment2(bbox_buttom_left, bbox_buttom_right), cRed);
        //     vis.addObject(GEOM_FADE25D::Segment2(bbox_top_right, bbox_buttom_right), cRed);

        //     //添加文字
        //     char object_info[128]={0};
        //     sprintf(object_info,"%s_%f",obj_iter->label.c_str(),obj_iter->prob);
        //     vis.addObject(GEOM_FADE25D::Label(bbox_top_left,std::string(object_info)),cRed);
        // }

        //绘制特征点
        const float radius = 3;
        for (size_t i = 0; i < mvKeys.size(); i++)
        {
            cv::Point2f pt = mvKeys[i].pt;
            GEOM_FADE25D::Point2 pfeature(pt.x, height - pt.y, 0);
            GEOM_FADE25D::Triangle2 *pTriangle = pdt_all->locate(pfeature);
            if (!pTriangle)
            {
                vis.addObject(GEOM_FADE25D::Circle2(pfeature, radius), cBlue);
            }
            else
            {
                //判断是否在一个点云簇上面
                GEOM_FADE25D::Point2 *p0 = pTriangle->getCorner(0);
                GEOM_FADE25D::Point2 *p1 = pTriangle->getCorner(1);
                GEOM_FADE25D::Point2 *p2 = pTriangle->getCorner(2);
                if (p0->getCustomIndex() != p1->getCustomIndex() || p0->getCustomIndex() != p2->getCustomIndex() || p1->getCustomIndex() != p2->getCustomIndex())
                {
                    // 特征点不在同一个目标上
                    vis.addObject(GEOM_FADE25D::Circle2(pfeature, radius), cBlue);
                }
                else
                {
                    // 特征点在同一个目标上
                    vis.addObject(GEOM_FADE25D::Circle2(pfeature, radius), cRed);
                }
            }
        }

        //绘制特征线段
        // for (size_t i = 0; i < mvKeyLines.size(); i++)
        // {
        //     cv::line_descriptor::KeyLine keyline = mvKeyLines[i];
        //     GEOM_FADE25D::Point2 s_line(keyline.startPointX, height - keyline.startPointY, 0);
        //     GEOM_FADE25D::Point2 e_line(keyline.endPointX, height - keyline.endPointY, 0);
        //     vis.addObject(GEOM_FADE25D::Segment2(s_line, e_line), cRed);
        // }

        vis.writeFile();
    }

    void LidarDepthExtration::PatchWorkGroundSegmentation(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_in, pcl::PointCloud<pcl::PointXYZI>::Ptr &ground_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr &obstacle_cloud, double time_takens)
    {
        mpPatchworkGroundSeg->estimate_ground(*lidar_in, *ground_cloud, *obstacle_cloud, time_takens);
        // std::cout << "pointcloud_in size: " << lidar_in->size()
        //           << "  ground_cloud size: " << ground_cloud->size()
        //           << "  obstacle_cloud size: " << obstacle_cloud->size() << std::endl;
    }

    void LidarDepthExtration::LidarDepthClustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pObstacleCloud,
                                                   std::unordered_map<uint16_t, depth_clustering::Cloud> &vLidarClusters, cv::Mat &rangeSegImage)
    {
        std::chrono::steady_clock::time_point t1_clustering = std::chrono::steady_clock::now();
        auto no_ground_cloud = depth_clustering::CloudFromPCL(pObstacleCloud);
        no_ground_cloud->InitProjection(*mpProjectionParams);

        int min_cluster_size = 20;
        int max_cluster_size = 100000;
        depth_clustering::Radians angle_tollerance = 10_deg;
        depth_clustering::ImageBasedClusterer<depth_clustering::LinearImageLabeler<>> clusterer(
            angle_tollerance, min_cluster_size, max_cluster_size);

        clusterer.SetDiffType(depth_clustering::DiffFactory::DiffType::ANGLES);
        clusterer.OnNewObjectReceived(*no_ground_cloud, vLidarClusters, rangeSegImage);

        std::chrono::steady_clock::time_point t2_clustering = std::chrono::steady_clock::now();
        double t_clustering_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2_clustering - t1_clustering).count();
    }

}
