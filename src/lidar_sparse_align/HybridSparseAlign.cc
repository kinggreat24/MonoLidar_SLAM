/*
 * @Author: kinggreat24
 * @Date: 2021-08-16 15:33:30
 * @LastEditTime: 2022-07-14 13:25:54
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /ORB_SLAM2/src/lidar_sparse_align/HybridSparseAlign.cc
 * 可以输入预定的版权声明、个性签名、空行等
 */

#include "lidar_sparse_align/HybridSparseAlign.h"

namespace ORB_SLAM2
{

    HybridSparseAlign::HybridSparseAlign(const vk::PinholeCamera *pinhole_model,
                                         const ORB_SLAM2::_tracker_t &photo_tracker_info,
                                         const ORB_SLAM2::_reproj_tracker_t &reproj_tracker_info)
        : photometric_tracker_info_(photo_tracker_info), reproj_tracker_info_(reproj_tracker_info), pinhole_model_(pinhole_model)
    {
        // 直接法相关的参数
        min_level_ = photometric_tracker_info_.min_level;
        max_level_ = photometric_tracker_info_.max_level;
        set_weightfunction();
    }

    HybridSparseAlign::~HybridSparseAlign() {}

    void HybridSparseAlign::set_weightfunction()
    {
        // 直接法权重函数，t-distribution
        if (photometric_tracker_info_.use_weight_scale)
        {
            switch (photometric_tracker_info_.scale_estimator_type)
            {
            case ScaleEstimatorType::TDistributionScale:
                photometric_scale_estimator_.reset(new TDistributionScaleEstimator());
                break;
            default:
                cerr << "HybridSparseAlign: Do not use photometric scale estimator." << endl;
            }
        }

        switch (photometric_tracker_info_.weight_function_type)
        {
        case WeightFunctionType::TDistributionWeight:
            photometric_weight_function_.reset(new TDistributionWeightFunction());
            break;
        default:
            cerr << "HybridSparseAlign: Do not use weight function." << endl;
        }

        // 间接法权重函数，gaussian-distribution, 使用huber-kerner
        if (reproj_tracker_info_.use_weight_scale)
        {
            switch (reproj_tracker_info_.scale_estimator_type)
            {
            case ScaleEstimatorType::MADScale:
                reproj_scale_estimator_.reset(new MADScaleEstimator());
                break;
            default:
                cerr << "HybridSparseAlign: Do not use reprojection scale estimator." << endl;
            }

            switch (reproj_tracker_info_.weight_function_type)
            {
            case WeightFunctionType::HuberWeight:
                reproj_weight_function_.reset(new HuberWeightFunction(std::sqrt(7.815)));
                break;
            default:
                cerr << "HybridSparseAlign: Do not use reprojection weight function." << endl;
            }
        }
    }

    // 具有深度信息的特征点，使用重投影误差
    bool HybridSparseAlign::hybrid_tracking(Frame *reference, Frame *current, std::vector<int> &feature_match, Sophus::SE3 &Tcl)
    {
        currFrameId_ = current->mnId;
        mvInvLevelSigma2 = reference->mvInvLevelSigma2;

        // 激光点处的patch
        ref_image_pyramid_ = reference->mvImgPyramid;
        pointcloud_ref_ = reference->pointcloud();
        cur_image_pyramid_ = current->mvImgPyramid;

        // 间接法相关
        mvMatches_ = feature_match;
        vKeyPointsRef = reference->mvKeysUn;
        vKeyPointsCur = current->mvKeysUn;
        vDepthRef = reference->mvDepth;

        // 参考帧特征点３Ｄ坐标
        vbInliner_ = std::vector<bool>(reference->N, true);
        vbValid_features_ = std::vector<bool>(reference->N, false);
        ref_keypoints_3d_ = std::vector<Eigen::Vector3d>(reference->N, Eigen::Vector3d(-1, -1, -1));

        for (size_t i = 0; i < reference->N; i++)
        {
            float depth = vDepthRef[i];
            if (depth <= 0)
                continue;

            if (mvMatches_[i] < 0)
                continue;

            cv::Point2f kp = vKeyPointsRef.at(i).pt;
            Eigen::Vector3d pt_3d_vec = depth * pinhole_model_->cam2world(Eigen::Vector2d(kp.x, kp.y));
            ref_keypoints_3d_[i] = pt_3d_vec;
            vbValid_features_[i] = true;
        }

        std::cout << "################    track with frame: " << reference->mnId << "  #########" << std::endl;
        //优化位置信息
        bool status = false;
        save_res_ = false;
        for (current_level_ = max_level_; current_level_ >= min_level_; current_level_--)
        {
            std::cout << "***********         track level: " << current_level_ << "      ***********" << std::endl;
            is_precomputed_ = false;
            stop_ = false;
            optimize(Tcl);
        }

        return status;
    }

    // 计算参考帧图像块
    void HybridSparseAlign::precompute_photometric_patches(cv::Mat &img, pcl::PointCloud<pcl::PointXYZI> &pointcloud, cv::Mat &patch_buf)
    {
        const int border = patch_halfsize_ + 2 + 2;
        const int stride = img.cols;
        const float scale = 1.0f / (1 << current_level_);

        // 将激光点投影到图像上
        std::vector<Eigen::Vector2d> uv_set;
        for (auto pt = pointcloud.begin(); pt != pointcloud.end(); pt++)
        {
            Eigen::Vector3d xyz(pt->x, pt->y, pt->z);
            Eigen::Vector2d uv = scale * pinhole_model_->world2cam(xyz);
            uv_set.push_back(uv);
        }

        patch_buf = cv::Mat(pointcloud.size(), pattern_length_, CV_32F);

        auto pc_iter = pointcloud.begin();
        size_t point_counter = 0;

        for (auto uv_iter = uv_set.begin(); uv_iter != uv_set.end(); ++uv_iter, ++pc_iter, ++point_counter)
        {
            Eigen::Vector2d &uv = *uv_iter;
            float u_f = uv(0);
            float v_f = uv(1);
            const int u_i = static_cast<int>(u_f);
            const int v_i = static_cast<int>(v_f);

            if (u_i - border < 0 || u_i + border > img.cols || v_i - border < 0 || v_i + border > img.rows || pc_iter->z <= 0.0)
            {
                float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;
                for (int i = 0; i < pattern_length_; ++i, ++patch_buf_ptr)
                    *patch_buf_ptr = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            const float subpix_u = u_f - u_i;
            const float subpix_v = v_f - v_i;
            const float w_tl = (1.0 - subpix_u) * (1.0 - subpix_v);
            const float w_tr = subpix_u * (1.0 - subpix_v);
            const float w_bl = (1.0 - subpix_u) * subpix_v;
            const float w_br = subpix_u * subpix_v;

            size_t pixel_counter = 0;

            float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;

            for (int i = 0; i < pattern_length_; ++i, ++pixel_counter, ++patch_buf_ptr)
            {
                int x = pattern_[i][0];
                int y = pattern_[i][1];

                float *img_ptr = (float *)img.data + (v_i + y) * stride + (u_i + x);
                *patch_buf_ptr = w_tl * img_ptr[0] + w_tr * img_ptr[1] + w_bl * img_ptr[stride] + w_br * img_ptr[stride + 1];
            }
        }
    }

    /**
     * 计算当前帧激光点patch块的像素值，梯度信息以及雅克比矩阵
     * @param img               当前帧图像
     * @param pointcloud_c      激光点
     * @param patch_buf         激光点对应的图像块
     * @param dI_buf output parameter of covariance of match
     * @param jacobian_buf whether to penalize matches further from the search center
     * @return strength of response
    */
    void HybridSparseAlign::precompute_photometric_patches(
        cv::Mat &img, pcl::PointCloud<pcl::PointXYZI> &pointcloud_c,
        cv::Mat &patch_buf,
        Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::ColMajor> &dI_buf,
        Eigen::Matrix<float, 6, Eigen::Dynamic, Eigen::ColMajor> &jacobian_buf)
    {
        const int border = patch_halfsize_ + 2 + 2;
        const int stride = img.cols;
        const float scale = 1.0f / (1 << current_level_);

        std::vector<Eigen::Vector2d> uv_set;
        for (auto pt = pointcloud_c.begin(); pt != pointcloud_c.end(); pt++)
        {
            Eigen::Vector3d xyz(pt->x, pt->y, pt->z);
            Eigen::Vector2d uv = scale * pinhole_model_->world2cam(xyz);
            uv_set.push_back(uv);
        }

        patch_buf = cv::Mat(pointcloud_c.size(), pattern_length_, CV_32F);

        auto pc_iter = pointcloud_c.begin();
        size_t point_counter = 0;

        // 计算每个patch的像素值以及对应的雅克比矩阵
        for (auto uv_iter = uv_set.begin(); uv_iter != uv_set.end(); ++uv_iter, ++pc_iter, ++point_counter)
        {
            Eigen::Vector2d &uv = *uv_iter;
            float u_f = uv(0);
            float v_f = uv(1);
            const int u_i = static_cast<int>(u_f);
            const int v_i = static_cast<int>(v_f);

            if (u_i - border < 0 || u_i + border > img.cols || v_i - border < 0 || v_i + border > img.rows || pc_iter->z <= 0.0)
            {
                float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;
                for (int i = 0; i < pattern_length_; ++i, ++patch_buf_ptr)
                    *patch_buf_ptr = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            const float subpix_u = u_f - u_i;
            const float subpix_v = v_f - v_i;
            const float w_tl = (1.0 - subpix_u) * (1.0 - subpix_v);
            const float w_tr = subpix_u * (1.0 - subpix_v);
            const float w_bl = (1.0 - subpix_u) * subpix_v;
            const float w_br = subpix_u * subpix_v;

            size_t pixel_counter = 0;

            float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;

            for (int i = 0; i < pattern_length_; ++i, ++pixel_counter, ++patch_buf_ptr)
            {
                int x = pattern_[i][0];
                int y = pattern_[i][1];

                float *img_ptr = (float *)img.data + (v_i + y) * stride + (u_i + x);
                *patch_buf_ptr = w_tl * img_ptr[0] + w_tr * img_ptr[1] + w_bl * img_ptr[stride] + w_br * img_ptr[stride + 1];

                // 计算雅克比矩阵
                // precompute image gradient
                float dx = 0.5f * ((w_tl * img_ptr[1] + w_tr * img_ptr[2] + w_bl * img_ptr[stride + 1] + w_br * img_ptr[stride + 2]) - (w_tl * img_ptr[-1] + w_tr * img_ptr[0] + w_bl * img_ptr[stride - 1] + w_br * img_ptr[stride]));
                float dy = 0.5f * ((w_tl * img_ptr[stride] + w_tr * img_ptr[1 + stride] + w_bl * img_ptr[stride * 2] + w_br * img_ptr[stride * 2 + 1]) - (w_tl * img_ptr[-stride] + w_tr * img_ptr[1 - stride] + w_bl * img_ptr[0] + w_br * img_ptr[1]));

                Matrix2x6 frame_jac;
                Eigen::Vector3f xyz(pc_iter->x, pc_iter->y, pc_iter->z);
                Frame::jacobian_xyz2uv(xyz, frame_jac);

                Eigen::Vector2f dI_xy(dx, dy);
                dI_buf.col(point_counter * pattern_length_ + i) = dI_xy;
                jacobian_buf.col(point_counter * pattern_length_ + pixel_counter) =
                    (dx * pinhole_model_->fx() * frame_jac.row(0) + dy * pinhole_model_->fy() * frame_jac.row(1)) / (1 << current_level_);
            }
        }
    }

    // 计算重投影误差以及雅克比矩阵
    double HybridSparseAlign::compute_reproject_residuals(const Sophus::SE3 &T_cur_ref)
    {
        reproj_errors_.clear();
        reproj_J_.clear();
        reproj_weight_.clear();
        reproj_weight_errors_.clear();
        reproj_errors_norm_.clear();

        n_reproject_measurement_ = 0;
        float chi2 = 0.0f;

        // double th_min = 0.0001;
        // double th_max = sqrt(7.815);

        for (size_t i = 0; i < vbValid_features_.size(); i++)
        {
            if (!vbValid_features_[i])
                continue;

            cv::Point2f uv_ref = vKeyPointsRef.at(i).pt;
            cv::Point2f uv_cur = vKeyPointsCur.at(mvMatches_[i]).pt;

            Eigen::Vector3d pt_3d_ref = ref_keypoints_3d_.at(i);
            Eigen::Vector3d pt_3d_cur = T_cur_ref.rotation_matrix() * pt_3d_ref + T_cur_ref.translation();
            Eigen::Vector2d uv_cur_proj = pinhole_model_->world2cam(pt_3d_cur);

            float invsigma = sqrt(mvInvLevelSigma2[vKeyPointsRef[i].octave]);

            // 重投影误差
            Eigen::Vector2d err = Eigen::Vector2d(uv_cur.x, uv_cur.y) - uv_cur_proj;
            reproj_errors_.push_back(err.cast<float>());
            reproj_weight_errors_.push_back(invsigma * err.cast<float>()); //加权误差
            reproj_errors_norm_.push_back(invsigma * err.norm());

            // 计算雅可比矩阵
            Matrix2x6 J = Matrix2x6::Identity();

            // 误差对P的导数
            double inv_z = 1.0 / pt_3d_cur.z();
            double inv_z2 = inv_z * inv_z;
            Eigen::Matrix<double, 2, 3> J_uv2p3d = Eigen::Matrix<double, 2, 3>::Zero();
            J_uv2p3d(0, 0) = pinhole_model_->fx() * inv_z;
            J_uv2p3d(0, 1) = 0.0;
            J_uv2p3d(0, 2) = -1 * pinhole_model_->fx() * pt_3d_cur.x() * inv_z2;
            J_uv2p3d(1, 0) = 0.0;
            J_uv2p3d(1, 1) = pinhole_model_->fy() * inv_z;
            J_uv2p3d(1, 2) = -1 * pinhole_model_->fy() * pt_3d_cur.y() * inv_z2;

            // P 对 \xi 的导数(sophus 平移在前，旋转在后)
            Eigen::Matrix<double, 3, 6> J_p3d2xi = Eigen::Matrix<double, 3, 6>::Zero();
            J_p3d2xi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            J_p3d2xi.block<3, 3>(0, 3) = -1 * Sophus::SO3::hat(pt_3d_cur);

            J = -1.0 * (J_uv2p3d * J_p3d2xi).cast<float>();

            // std::cout<<"*******************   J_uv2p3d: "<<std::endl<<J_uv2p3d.matrix()<<"J_p3d2xi: "<<J_p3d2xi.matrix()<<"J: "<<J.matrix()<<std::endl;

            reproj_J_.push_back(J);

            n_reproject_measurement_++;
        }

        // // estimate scale of the residuals
        // double s_p = 1.0;
        // s_p = reproj_scale_estimator_->compute(reproj_errors_norm_);
        // if (s_p < th_min)
        //     s_p = th_min;
        // if (s_p > th_max)
        //     s_p = th_max;
        // std::cout << "sp: " << s_p << std::endl;

        if (iter_ == 0)
        {
            reprojection_sigma_ = calculate_visual_mu_delta(reproj_weight_errors_);
            std::cout << "visual_delta: " << reprojection_sigma_ << std::endl;
        }

        // // if employing robust cost function
        for (size_t i = 0; i < reproj_errors_norm_.size(); i++)
        {
            double r = reproj_errors_norm_.at(i);
            double w = reproj_weight_function_->weight(r);
            reproj_weight_.push_back(w);

            chi2 += r * r * w;
        }

        return chi2 / n_reproject_measurement_;
    }

    double HybridSparseAlign::compute_photometric_residuals(const Sophus::SE3 &T_cur_ref)
    {
        photometric_errors_.clear();
        photometric_J_.clear();
        photometric_weight_.clear();

        if (!is_precomputed_)
        {
            // 每一层金字塔的第一次迭代，需要重新计算参考帧图像块的信息
            if (!ref_patch_buf_.empty())
                ref_patch_buf_.release();

            cv::Mat reference_img = ref_image_pyramid_[current_level_].clone();
            precompute_photometric_patches(reference_img, pointcloud_ref_, ref_patch_buf_);

            // std::cout << "Init dI_buf & jacobian_buf" << std::endl;
            dI_buf_.resize(Eigen::NoChange, ref_patch_buf_.rows * pattern_length_);
            dI_buf_.setZero();
            jacobian_buf_.resize(Eigen::NoChange, ref_patch_buf_.rows * pattern_length_);
            jacobian_buf_.setZero();

            is_precomputed_ = true;
        }

        // // TODO:把这一部分封装成一个函数，方便实现多线程计算
        // // 计算激光点对应的图像块以及雅克比矩阵
        cv::Mat current_img = cur_image_pyramid_[current_level_].clone();
        pcl::PointCloud<pcl::PointXYZI> pointcloud_cur;
        pcl::transformPointCloud(pointcloud_ref_, pointcloud_cur, T_cur_ref.matrix());
        precompute_photometric_patches(current_img, pointcloud_cur, cur_patch_buf_, dI_buf_, jacobian_buf_);

        // // 计算直接法权重
        cv::Mat errors = cv::Mat(pointcloud_cur.size(), pattern_length_, CV_32F);
        errors = cur_patch_buf_ - ref_patch_buf_;
        // scale_ = scale_estimator_->compute(errors);

        float chi2 = 0.0f;
        n_photometric_measurement_ = 0;

        float *errors_ptr = errors.ptr<float>();
        float *ref_patch_buf_ptr = ref_patch_buf_.ptr<float>();
        float *cur_patch_buf_ptr = cur_patch_buf_.ptr<float>();

        float IiIj = 0.0f;
        float IiIi = 0.0f;
        float sum_Ii = 0.0f;
        float sum_Ij = 0.0f;

        for (int i = 0; i < errors.size().area(); ++i, ++errors_ptr, ++ref_patch_buf_ptr, ++cur_patch_buf_ptr)
        {

            float &res = *errors_ptr;

            float &Ii = *ref_patch_buf_ptr;
            float &Ij = *cur_patch_buf_ptr;

            if (std::isfinite(res))
            {
                n_photometric_measurement_++;

                Vector6 J(jacobian_buf_.col(i));

                photometric_errors_.push_back(res);
                photometric_J_.push_back(-1.0 * J);

                IiIj += Ii * Ij;
                IiIi += Ii * Ii;
                sum_Ii += Ii;
                sum_Ij += Ij;
            }
        }

        affine_a_ = IiIj / IiIi;
        affine_b_ = (sum_Ij - affine_a_ * sum_Ii) / n_photometric_measurement_;

        vector<float> sorted_errors;
        sorted_errors.resize(photometric_errors_.size());
        copy(photometric_errors_.begin(), photometric_errors_.end(), sorted_errors.begin());
        sort(sorted_errors.begin(), sorted_errors.end());

        float median_mu = sorted_errors[sorted_errors.size() / 2];

        std::vector<float> absolute_res_error;
        for (auto error : photometric_errors_)
        {
            absolute_res_error.push_back(fabs(error - median_mu));
        }
        sort(absolute_res_error.begin(), absolute_res_error.end());
        float median_abs_deviation = 1.4826 * absolute_res_error[absolute_res_error.size() / 2];

        photometric_weight_errors_.clear();
        for (auto error : photometric_errors_)
        {
            float weight = 1.0;
            weight = photometric_weight_function_->weight((error - median_mu) / median_abs_deviation);
            photometric_weight_.push_back(weight);

            photometric_weight_errors_.push_back(weight * error);

            chi2 += error * error * weight;
        }

        // std::cout << "median_abs_deviation: " << median_abs_deviation << std::endl;
        if (iter_ == 0)
        {
            photometric_sigma_ = calculate_1dof_mu_delta(photometric_weight_errors_);
            std::cout << "photometric delta: " << photometric_sigma_ << std::endl;
        }

        return chi2 / n_photometric_measurement_;
    }

    void HybridSparseAlign::max_level(int level) { max_level_ = level; }

    double HybridSparseAlign::build_LinearSystem(Sophus::SE3 &model)
    {
        // 直接法计算运动
        double res = 0.0;
        // res = compute_photometric_residuals(model);

        H_.setZero();
        Jres_.setZero();

        // for (int i = 0; i < photometric_errors_.size(); ++i)
        // {
        //     float &res = photometric_errors_[i];
        //     Vector6 &J = photometric_J_[i];
        //     float &weight = photometric_weight_[i];

        //     H_.noalias() += J * J.transpose() * weight;
        //     Jres_.noalias() -= J * res * weight;
        // }

        // 间接法计算运动信息
        res = compute_reproject_residuals(model);

        double alpha = 0.0001;

        double alpha1 = 0.0;
        // if (iter_ == 0)
        // {
        //     alpha1 = 0.1 * photometric_sigma_ / reprojection_sigma_;
        //     std::cout << "alpha1: " << alpha1 << std::endl;
        // }

        for (int i = 0; i < reproj_errors_.size(); ++i)
        {
            Vector2 &res = reproj_errors_[i];
            Matrix2x6 &J = reproj_J_[i];
            float &weight = reproj_weight_[i];

            H_.noalias() += J.transpose() * J * weight;
            Jres_.noalias() -= J.transpose() * res * weight;
        }

        // if (!save_res_)
        // {
        //     saveResidual2file("/home/kinggreat24/pc", "photometric", currFrameId_, photometric_errors_);
        //     saveResidual2file("/home/kinggreat24/pc", "geometric", currFrameId_, reproj_errors_norm_);
        //     save_res_ = true;
        // }

        return res;
    }

    // implementation for LSQNonlinear class
    void HybridSparseAlign::update(const ModelType &old_model, ModelType &new_model)
    {
        Eigen::Matrix<double, 6, 1> update_;
        for (int i = 0; i < 6; i++)
            update_[i] = x_[i];
        new_model = Sophus::SE3::exp(update_) * old_model;
    }

    void HybridSparseAlign::removeOutliers(Eigen::Matrix4d DT)
    {

        //TODO: if not usig mad stdv, use just a fixed threshold (sqrt(7.815)) to filter outliers (with a single for loop...)

        // // point features
        // vector<double> res_p;
        // res_p.reserve(matched_pt.size());
        // int iter = 0;
        // for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
        // {
        //     // projection error
        //     Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
        //     Vector2d pl_proj = cam->projection( P_ );
        //     res_p.push_back( ( pl_proj - (*it)->pl_obs ).norm() * sqrt((*it)->sigma2) );
        //     //res_p.push_back( ( pl_proj - (*it)->pl_obs ).norm() );
        // }
        // // estimate robust parameters
        // double p_stdv, p_mean, inlier_th_p;
        // vector_mean_stdv_mad( res_p, p_mean, p_stdv );
        // inlier_th_p = Config::inlierK() * p_stdv;
        // //inlier_th_p = sqrt(7.815);
        // //cout << endl << p_mean << " " << p_stdv << "\t" << inlier_th_p << endl;
        // // filter outliers
        // iter = 0;
        // for( auto it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
        // {
        //     if( (*it)->inlier && fabs(res_p[iter]-p_mean) > inlier_th_p )
        //     {
        //         (*it)->inlier = false;
        //         n_inliers--;
        //         n_inliers_pt--;
        //     }
        // }
    }

    void HybridSparseAlign::saveResidual2file(const std::string &file_dir, const std::string &err_type, const unsigned long frame_id, const std::vector<float> &residuals)
    {
        char file_name[128] = {0};
        sprintf(file_name, "%s/%d_%s.txt", file_dir.c_str(), frame_id, err_type.c_str());

        // //边缘点误差分布
        std::ofstream ofs(file_name);
        for (size_t i = 0; i < residuals.size(); i++)
        {
            if (residuals[i] < 0.000001)
                continue;

            // double edge_res = corner_residuals[3 * i + 0] * corner_residuals[3 * i + 0] +
            //                   corner_residuals[3 * i + 1] * corner_residuals[3 * i + 1] +
            //                   corner_residuals[3 * i + 2] * corner_residuals[3 * i + 2];
            // ofs << edge_res << " " << std::endl;

            ofs << residuals[i] << std::endl;
        }
        ofs.flush();
        ofs.close();
    }
}