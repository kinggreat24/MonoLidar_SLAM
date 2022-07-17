/*
 * @Author: kinggreat24
 * @Date: 2021-08-16 15:33:44
 * @LastEditTime: 2022-07-14 10:24:30
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /ORB_SLAM2/include/lidar_sparse_align/HybridSparseAlign.h
 * 可以输入预定的版权声明、个性签名、空行等
 */

#ifndef SEMI_DIRECT_LIDAR_ALIGN_H
#define SEMI_DIRECT_LIDAR_ALIGN_H

#include <iostream>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <sophus/se3.h>

#include "lidar_sparse_align/WeightFunction.h"
#include "lidar_sparse_align/LSQNonlinear.hpp"

#include "Frame.h"
#include "KeyFrame.h"

#include <vikit/pinhole_camera.h>
#include <vikit/vision.h>

namespace ORB_SLAM2
{

    // 利用匹配的图像点和直接法联合进行里程计的计算
    class HybridSparseAlign : public LSQNonlinearGaussNewton<6, Sophus::SE3>
    {
        static const int patch_halfsize_ = 2;
        static const int patch_size_ = 2 * patch_halfsize_;
        static const int patch_area_ = patch_size_ * patch_size_;

        static const int pattern_length_ = 8;
        int pattern_[8][2] = {{0, 0}, {2, 0}, {1, 1}, {0, -2}, {-1, -1}, {-2, 0}, {-1, 1}, {0, 2}};

    public:
        HybridSparseAlign(const vk::PinholeCamera *pinhole_model, const ORB_SLAM2::_tracker_t &tracker_info, const ORB_SLAM2::_reproj_tracker_t &reproj_tracker_info);
        ~HybridSparseAlign();

        //
        bool hybrid_tracking(Frame *reference, Frame *current, std::vector<int> &feature_match, Sophus::SE3 &transformation);

    private:
        int currFrameId_;
        int current_level_;
        int min_level_;
        int max_level_;

        bool display_;   //!< display residual image.
        cv::Mat resimg_; // residual image.

        const vk::PinholeCamera *pinhole_model_;

        Sophus::SE3 Tji_;

        pcl::PointCloud<pcl::PointXYZI> pointcloud_ref_; // 参考帧激光点云(采样出来梯度比较大的点)
        std::vector<cv::Mat> ref_image_pyramid_;         // 参考帧图像金字塔
        std::vector<cv::Mat> cur_image_pyramid_;         // 当前帧图像金字塔

        std::vector<Eigen::Vector3d> ref_keypoints_3d_; //参考帧特征点３Ｄ坐标
        std::vector<cv::KeyPoint> vKeyPointsRef;
        std::vector<cv::KeyPoint> vKeyPointsCur; //当前帧特征点
        std::vector<float> vDepthRef;
        std::vector<int> mvMatches_;
        std::vector<bool> vbValid_features_; //是否为外点
        std::vector<bool> vbInliner_;        //是否为外点

        bool is_precomputed_; // 是否已经计算过梯度
        cv::Mat ref_patch_buf_, cur_patch_buf_;
        Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::ColMajor> dI_buf_;
        Eigen::Matrix<float, 6, Eigen::Dynamic, Eigen::ColMajor> jacobian_buf_;

        // 光度误差对应的雅克比矩阵、误差、权重
        int n_photometric_measurement_;
        vector<float> photometric_errors_;
        vector<float> photometric_weight_errors_; //加权误差
        vector<Vector6> photometric_J_;
        vector<float> photometric_weight_;
        float affine_a_;
        float affine_b_;

        // 重投影误差对应的加权后误差、雅克比矩阵以及权重
        int n_reproject_measurement_;
        vector<Vector2> reproj_errors_;
        vector<Vector2> reproj_weight_errors_; //加权误差
        vector<float> reproj_errors_norm_;
        vector<Matrix2x6> reproj_J_;
        vector<float> reproj_weight_;

        void precompute_photometric_patches(cv::Mat &img, pcl::PointCloud<pcl::PointXYZI> &pointcloud,
                                            cv::Mat &patch_buf);

        // 直接法梯度计算
        void precompute_photometric_patches(cv::Mat &img, pcl::PointCloud<pcl::PointXYZI> &pointcloud_c,
                                            cv::Mat &patch_buf,
                                            Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::ColMajor> &dI_buf,
                                            Eigen::Matrix<float, 6, Eigen::Dynamic, Eigen::ColMajor> &jacobian_buf);

        double compute_reproject_residuals(const Sophus::SE3 &T_cur_ref);

        double compute_photometric_residuals(const Sophus::SE3 &T_cur_ref);

        // implementation for LSQNonlinear class
        virtual void update(const ModelType &old_model, ModelType &new_model);

        void removeOutliers(Eigen::Matrix4d DT);

        void saveResidual2file(const std::string &file_dir, const std::string &err_type, const unsigned long frame_id, const std::vector<float> &residuals);

    public:
        // weight function
        ORB_SLAM2::_tracker_t photometric_tracker_info_;
        ORB_SLAM2::_reproj_tracker_t reproj_tracker_info_;
        bool use_weight_scale_;
        bool save_res_;
        float scale_;

        float photometric_sigma_;
        float reprojection_sigma_;

        std::vector<float> mvInvLevelSigma2;

        // 直接法相关的权重函数
        std::shared_ptr<ScaleEstimator> photometric_scale_estimator_;
        std::shared_ptr<WeightFunction> photometric_weight_function_;

        // 间接法相关的权重函数
        std::shared_ptr<ScaleEstimator> reproj_scale_estimator_;
        std::shared_ptr<WeightFunction> reproj_weight_function_;

        void set_weightfunction();
        void max_level(int level);

    protected:
        virtual double build_LinearSystem(Sophus::SE3 &model);

        inline double calculate_visual_mu_delta(const std::vector<Vector2> &residuals)
        {
            // assert(residuals.size() % 2 == 0);
            int nm = residuals.size();

            double mu_err_u = 0.0;
            double mu_err_v = 0.0;
            for (int i = 0; i < nm; i++)
            {
                mu_err_u += residuals.at(i).x();
                mu_err_v += residuals.at(i).y();
            }
            mu_err_u /= nm;
            mu_err_v /= nm;

            //计算方差
            double delta_err_u = 0.0;
            double delta_err_v = 0.0;
            for (int i = 0; i < nm; i++)
            {
                delta_err_u += std::pow(residuals.at(i).x() - mu_err_u, 2);
                delta_err_v += std::pow(residuals.at(i).y() - mu_err_v, 2);
            }

            delta_err_u /= (nm - 1);
            delta_err_v /= (nm - 1);

            return std::sqrt((delta_err_u + delta_err_v) / 2.0);
        }

        inline double calculate_1dof_mu_delta(const std::vector<float> &residuals)
        {
            int nm = residuals.size();
            double mu_err = 0.0;
            for (int i = 0; i < nm; i++)
            {
                mu_err += residuals.at(i);
            }
            mu_err /= nm;

            //计算方差
            double delta_err = 0.0;
            for (int i = 0; i < nm; i++)
            {
                delta_err += std::pow(residuals.at(i) - mu_err, 2);
            }

            delta_err /= (nm - 1);

            return std::sqrt(delta_err);
        }
    };

}

#endif //SEMI_DIRECT_LIDAR_ALIGN_H