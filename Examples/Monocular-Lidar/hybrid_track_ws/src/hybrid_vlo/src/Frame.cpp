/*
 * @Author: your name
 * @Date: 2021-09-20 20:20:18
 * @LastEditTime: 2022-07-12 09:42:36
 * @LastEditors: kinggreat24
 * @Description: In User Settings Edit
 * @FilePath: /ORB_SLAM2/Examples/Monocular-Lidar/hybrid_track_ws/src/hybrid_vlo/src/Frame.cpp
 */

#include "Frame.h"
#include <vikit/patch_score.h>
#include <vikit/vision.h>

#include <pcl/common/transforms.h>
#include <pcl-1.7/pcl/io/pcd_io.h>

namespace hybrid_vlo
{
    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;

    Frame::Frame(cv::Mat &imL, cv::Mat &imR, std::string &lidar_file, ORB_SLAM2::ORBextractor *pORBExtractor, vk::PinholeCamera *pinhole_cam, int nlevles,
                 Eigen::Matrix4d &lidar2camera_matrix, float bf, bool depth_flag)
    {
        // Frame ID
        mnId = nNextId++;

        std::cout << "construct frame: " << mnId << std::endl;

        mpPinholeCam_ = pinhole_cam;
        mpORBExtractor = pORBExtractor;

        // 激光到相机的外参变化
        mLidar2camMatrix = lidar2camera_matrix;

        // 创建图像金字塔
        std::cout << "build image pyramid" << std::endl;
        int mnTrackLevels = nlevles;
        cv::Mat original_img_ = imL.clone();
        original_img_.convertTo(original_img_, CV_32FC1, 1.0 / 255);
        create_image_pyramid(original_img_, mnTrackLevels, mvImgPyramid_);

        // 读取激光雷达数据
        std::cout << "read lidar points" << std::endl;
        pcl::PointCloud<PointType>::Ptr pLidarPointsCloud(new pcl::PointCloud<PointType>());
        ReadPointCloud(lidar_file, pLidarPointsCloud, true);
        mpLidarPointCloudCamera.reset(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*pLidarPointsCloud, *mpLidarPointCloudCamera, mLidar2camMatrix);

        PointSampling();

        // 提取图像特征
        ExtractORB(imL);

        N = mvKeys.size();

        // 双目生成深度
        // if (depth_flag)
        // {
        //     // 传入的图像为计算的视差图
        //     std::cout << "compute feature depth from disparity image" << std::endl;
        //     int nvalid = ComputeStereoDepth(imR, bf);
        //     std::cout << "valid feature points: " << nvalid << std::endl;
        // }
        // else
        // {
        //     std::cout << "compute image disparity" << std::endl;
        //     mDisparityImage = ComputeStereoDisparity(imL, imR);
        //     std::cout << "compute feature depth" << std::endl;
        //     int nvalid = ComputeStereoDepth(mDisparityImage, bf);
        //     std::cout << "valid feature points: " << nvalid << std::endl;
        // }

        if (mbInitialComputations)
        {
            ComputeImageBounds(imL);
            mbInitialComputations = false;
        }
    }

    Frame::~Frame()
    {
        std::cout << "###########            Frame destory             ##############" << std::endl;
        mvImgPyramid_.clear();

        if (mpMagLidarPointCloud)
            mpMagLidarPointCloud.reset(new pcl::PointCloud<PointType>());
        if (mpLidarPointCloudCamera)
            mpLidarPointCloudCamera.reset(new pcl::PointCloud<PointType>());
    }

    void Frame::ExtractORB(cv::Mat &imGray)
    {
        (*mpORBExtractor)(imGray, cv::Mat(), mvKeys, mDescriptor);
    }

    void Frame::PointSampling()
    {
        int num_bucket_size = 20;

        std::vector<std::pair<float, PointType>> mag_point_bucket;
        mag_point_bucket.reserve(num_bucket_size);

        int num_out_points = 0;

        mpMagLidarPointCloud.reset(new pcl::PointCloud<PointType>());

        for (auto iter = mpLidarPointCloudCamera->begin(); iter != mpLidarPointCloudCamera->end(); iter++)
        {
            pcl::PointXYZI pt = *iter;
            if (pt.z < 0)
                continue;

            Eigen::Vector2d uv = mpPinholeCam_->world2cam(Eigen::Vector3d(pt.x, pt.y, pt.z));
            if (mpPinholeCam_->isInFrame(uv.cast<int>(), 0))
            {
                int u = static_cast<int>(uv(0));
                int v = static_cast<int>(uv(1));

                cv::Mat img = mvImgPyramid_.at(0);
                float dx = 0.5f * (img.at<float>(v, u + 1) - img.at<float>(v, u - 1));
                float dy = 0.5f * (img.at<float>(v + 1, u) - img.at<float>(v - 1, u));

                std::pair<float, PointType> mag_point;
                mag_point = std::make_pair((dx * dx + dy * dy), (*iter));

                mag_point_bucket.push_back(mag_point);
                if (mag_point_bucket.size() == num_bucket_size)
                {

                    float max = -1;
                    int idx;
                    for (int i = 0; i < mag_point_bucket.size(); ++i)
                    {
                        if (mag_point_bucket[i].first > max)
                        {
                            max = mag_point_bucket[i].first;
                            idx = i;
                        }
                    }

                    if (max > (6.25 / (255.0 * 255.0))) // 16.25
                        mpMagLidarPointCloud->push_back(mag_point_bucket[idx].second);

                    mag_point_bucket.clear();
                }
            }
        }
    }

    cv::Mat Frame::ComputeStereoDisparity(const cv::Mat &left_image, const cv::Mat &right_image,
                                          const int disp_size, const bool subpixel,
                                          const int output_depth)
    {
        cv::Mat I1 = left_image.clone();
        cv::Mat I2 = right_image.clone();

        const int width = I1.cols;
        const int height = I1.rows;

        const int input_depth = I1.type() == CV_8U ? 8 : 16;
        const int input_bytes = input_depth * width * height / 8;
        const int output_bytes = output_depth * width * height / 8;

        const sgm::StereoSGM::Parameters params{10, 120, 0.95f, subpixel};

        sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA, params);

        // 输出
        cv::Mat disparity(height, width, CV_16S);
        cv::Mat disparity_32f;

        device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);

        cudaMemcpy(d_I1.data, I1.data, input_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_I2.data, I2.data, input_bytes, cudaMemcpyHostToDevice);

        const auto t1 = std::chrono::system_clock::now();

        sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
        cudaDeviceSynchronize();

        const auto t2 = std::chrono::system_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        const double fps = 1e6 / duration;

        std::cout << "SGM time: " << duration * 1e-3 << "ms"
                  << "   fps: " << fps << std::endl;

        cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

        // draw results
        if (I1.type() != CV_8U)
        {
            cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
            I1.convertTo(I1, CV_8U);
        }

        disparity.convertTo(disparity_32f, CV_32F, subpixel ? 1. / sgm::StereoSGM::SUBPIXEL_SCALE : 1);

        return disparity_32f;
    }

    // 利用视差图计算特征点的深度
    void Frame::Compute3dKeyPoints(const std::vector<cv::Point2f> &vKeyPoints, const cv::Mat &disparity, const CameraParameters &camera, std::vector<cv::Point3f> &pt_3ds)
    {
        CoordinateTransform tf(camera);

        pt_3ds.clear();
        pt_3ds.resize(vKeyPoints.size(), cv::Point3f(-1, -1, -1));

        for (size_t i = 0; i < vKeyPoints.size(); i++)
        {
            const float d = disparity.at<float>(vKeyPoints.at(i).y, vKeyPoints.at(i).x);

            if (d > 0)
                pt_3ds[i] = tf.imageToWorld(cv::Point(vKeyPoints.at(i).x, vKeyPoints.at(i).y), d);
        }
    }

    void Frame::ComputeStereoPointCloud(const cv::Mat &image_gray, const cv::Mat &disparity, const CameraParameters &camera, bool subpixeled,
                                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_3d_ptr)
    {
        CV_Assert(disparity.type() == CV_32F);

        CoordinateTransform tf(camera);

        for (int y = 0; y < disparity.rows; y++)
        {
            for (int x = 0; x < disparity.cols; x++)
            {
                const float d = disparity.at<float>(y, x);
                if (d > 0)
                {
                    pcl::PointXYZRGB pt_color;
                    cv::Point3f pt_3d = tf.imageToWorld(cv::Point(x, y), d);
                    pt_color.x = pt_3d.x;
                    pt_color.y = pt_3d.y;
                    pt_color.z = pt_3d.z;

                    pt_color.r = image_gray.at<uchar>(y, x);
                    pt_color.g = image_gray.at<uchar>(y, x);
                    pt_color.b = image_gray.at<uchar>(y, x);
                    points_3d_ptr->push_back(pt_color);
                }
            }
        }
        points_3d_ptr->width = points_3d_ptr->size();
        points_3d_ptr->height = 1;
    }

    int Frame::ComputeStereoDepth(const cv::Mat &disparity, const float bf)
    {
        int n = 0;
        mvDepths.resize(mvKeys.size(), -1);
        for (size_t i = 0; i < mvKeys.size(); i++)
        {
            cv::KeyPoint kpt = mvKeys.at(i);
            float d = vk::interpolateMat_32f(disparity, kpt.pt.x, kpt.pt.y);
            if (d <= 0)
                continue;

            float depth_ = bf / d;

            // if (depth_ > 100.0)
            //     depth_ = -1.0;
            mvDepths.at(i) = depth_;

            n++;
        }
        return n;
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft)
    {
        if (mpPinholeCam_->d0() != 0.0)
        {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            cv::Mat mK = cv::Mat::zeros(3, 3, CV_32F);
            mK.at<float>(0, 0) = mpPinholeCam_->fx();
            mK.at<float>(1, 1) = mpPinholeCam_->fy();
            mK.at<float>(0, 2) = mpPinholeCam_->cx();
            mK.at<float>(1, 2) = mpPinholeCam_->cy();

            cv::Mat mDistCoef = cv::Mat::zeros(1, 5, CV_32F);
            mDistCoef.at<float>(0) = mpPinholeCam_->d0();
            mDistCoef.at<float>(1) = mpPinholeCam_->d1();
            mDistCoef.at<float>(2) = mpPinholeCam_->d2();
            mDistCoef.at<float>(3) = mpPinholeCam_->d3();
            mDistCoef.at<float>(4) = mpPinholeCam_->d4();

            // Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);

            mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
        }
        else
        {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

    // 特征深度拟合
    GEOM_FADE25D::Fade_2D *Frame::CreateTerrain(pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_cam)
    {
        GEOM_FADE25D::Fade_2D *pDt = new GEOM_FADE25D::Fade_2D();

        //将点云转到相机坐标空间
        std::vector<GEOM_FADE25D::Point2> vPoints;
        for (int i = 0; i < pc_cam->size(); i++)
        {
            pcl::PointXYZI pt = pc_cam->points.at(i);
            if (pt.z < 0)
                continue;

            Eigen::Vector2d uv_ref = mpPinholeCam_->world2cam(Eigen::Vector3d(pt.x, pt.y, pt.z));
            if (!mpPinholeCam_->isInFrame(uv_ref.cast<int>(), 0))
                continue;

            float u = uv_ref.x();
            float v = uv_ref.y();

            GEOM_FADE25D::Point2 p(u, mnMaxY - v, pt.z);
            p.setCustomIndex(0);

            vPoints.push_back(p);
        }
        pDt->insert(vPoints);
        return pDt;
    }

    // GEOM_FADE25D::Fade_2D *Frame::CreateTerrain(std::unordered_map<uint16_t, depth_clustering::Cloud> &clusters)
    // {
    //     GEOM_FADE25D::Fade_2D *pDt = new GEOM_FADE25D::Fade_2D();

    //     std::vector<GEOM_FADE25D::Point2> vPoints;
    //     int cluster_idx = 0;
    //     for (auto cluster_iter = clusters.begin(); cluster_iter != clusters.end(); cluster_iter++)
    //     {
    //         //
    //         pcl::PointCloud<pcl::PointXYZI>::Ptr pc_cluster = cluster_iter->second.ToPcl(cluster_iter->first);

    //         //将点云转到相机坐标空间
    //         pcl::PointCloud<pcl::PointXYZI>::Ptr pc_cluster_cam(new pcl::PointCloud<pcl::PointXYZI>());
    //         pcl::transformPointCloud(*pc_cluster, *pc_cluster_cam, mLidar2CameraExtric);

    //         for (int i = 0; i < pc_cluster_cam->size(); i++)
    //         {
    //             pcl::PointXYZI pt = pc_cluster_cam->points.at(i);
    //             if (pt.z < 0)
    //                 continue;

    //             Eigen::Vector2d uv_ref = mpPinholeCamera->world2cam(Eigen::Vector3d(pt.x,pt.y,pt.z));
    //             if(!mpPinholeCamera->isInFrame(uv_ref.cast<int>(),0))
    //                 continue;

    //             //投影到图像上
    //             float u = uv_ref.x();
    //             float v = uv_ref.y();

    //             // if (u >= mnMaxX || u < mnMinX || v >= mnMaxY || v < mnMinY)
    //             //     continue;

    //             GEOM_FADE25D::Point2 p(u, mnMaxY - v, pt.z);
    //             p.setCustomIndex(cluster_idx);
    //             vPoints.push_back(p);
    //         }
    //         cluster_idx++;
    //     }
    //     // GEOM_FADE25D::EfficientModel em(vPoints);
    //     // vPoints.clear();
    //     // double maxError(.1);
    //     // em.extract(maxError,vPoints);
    //     pDt->insert(vPoints);
    //     return pDt;
    // }

    float Frame::DepthFitting(GEOM_FADE25D::Fade_2D *pdt, const cv::Point2f &pt)
    {
        GEOM_FADE25D::Point2 pfeature(pt.x, mnMaxY - pt.y, 0);
        GEOM_FADE25D::Triangle2 *pTriangle = pdt->locate(pfeature);

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
                return -1.0;
        }
        else
            return -1.0;

        normalVector = AB.cross(AC);
        normalVector.normalize();

        Eigen::Vector3d AP(pfeature.x() - pA->x(), pfeature.y() - pA->y(), pfeature.z() - pA->z());
        float depth = -(normalVector(0) * AP(0) + normalVector(1) * AP(1)) / normalVector(2) + pA->z();
        return depth;
    }

    void Frame::DrawObjectDenuary(GEOM_FADE25D::Fade_2D *pdt_all,
                                  const std::vector<cv::KeyPoint> &mvKeys, std::string file_name)
    {
        GEOM_FADE25D::Visualizer2 vis(file_name);

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
        GEOM_FADE25D::Point2 p_top_left(0, mnMaxY, 0);
        GEOM_FADE25D::Point2 p_top_right(mnMinX, mnMaxY, 0);
        GEOM_FADE25D::Point2 p_buttom_left(0, 0, 0);
        GEOM_FADE25D::Point2 p_buttom_right(mnMinX, 0, 0);
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
            GEOM_FADE25D::Point2 pfeature(pt.x, mnMaxY - pt.y, 0);
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
                    vis.addObject(GEOM_FADE25D::Circle2(pfeature, radius), cBlue);
                else
                {
                    vis.addObject(GEOM_FADE25D::Circle2(pfeature, radius), cRed);
                    //深度拟合
                }
            }
        }

        vis.writeFile();
    }

    void Frame::ShowPointClouds(const pcl::PointCloud<pcl::PointXYZI>::Ptr &mpLidarPointCloud,
                                cv::Mat &image_out, size_t num_level)
    {
        cv::Mat img = mvImgPyramid_[num_level];
        if (img.channels() == 1)
        {
            cvtColor(img, image_out, cv::COLOR_GRAY2BGR);
        }
        else
        {
            img.copyTo(image_out);
        }

        const float scale = 1.0f / (1 << num_level);

        int n = 0;
        for (auto iter = mpLidarPointCloud->begin(); iter != mpLidarPointCloud->end(); ++iter)
        {
            Eigen::Vector3d xyz_ref(iter->x, iter->y, iter->z);

            if (iter->z <= 0)
                continue;

            Eigen::Vector2d uv_ref;
            uv_ref = mpPinholeCam_->world2cam(xyz_ref) * scale;

            if (!mpPinholeCam_->isInFrame(uv_ref.cast<int>(), 0))
                continue;

            const float u_ref_f = uv_ref(0);
            const float v_ref_f = uv_ref(1);
            const int u_ref_i = static_cast<int>(u_ref_f);
            const int v_ref_i = static_cast<int>(v_ref_f);

            float v_min = 1.0;
            float v_max = 50.0;
            float dv = v_max - v_min;
            float v = xyz_ref[2];
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

            cv::circle(image_out, cv::Point(u_ref_i, v_ref_i), 1.0, cv::Scalar(r, g, b), -1);
        }
        image_out.convertTo(image_out, CV_8UC3, 255);
    }

    void Frame::ShowFeaturePoints(cv::Mat &image_out)
    {
        if (image_out.channels() == 1)
            cv::cvtColor(image_out, image_out, CV_GRAY2BGR);

        const float r = 5.0;
        for (size_t i = 0; i < mvKeys.size(); i++)
        {
            cv::Point2f pt1, pt2;
            pt1.x = mvKeys[i].pt.x - r;
            pt1.y = mvKeys[i].pt.y - r;
            pt2.x = mvKeys[i].pt.x + r;
            pt2.y = mvKeys[i].pt.y + r;

            cv::rectangle(image_out, pt1, pt2, cv::Scalar(0, 255, 0));
            cv::circle(image_out, mvKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);
        }
    }

    cv::Scalar Frame::randomColor(int64 seed)
    {
        cv::RNG rng(seed);
        int icolor = (unsigned int)rng;
        return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
    }

} // end of namespace