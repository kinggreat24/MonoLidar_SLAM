/*
 * @Author: your name
 * @Date: 2021-09-21 22:35:20
 * @LastEditTime: 2021-09-28 21:48:49
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /NLS_examples/src/lidar_sparse_align/feature_align.cc
 */
#include "lidar_sparse_align/FeatureAlign.h"

namespace hybrid_vlo
{

    FeatureAlign::FeatureAlign(const vk::PinholeCamera *pinhole_model, const _tracker_t &tracker_info)
        : pinhole_model_(pinhole_model), tracker_info_(tracker_info)
    {
        tracker_info_.set_scale_estimator_type();
        tracker_info_.set_weight_function_type();
        set_weightfunction();

        std::cout << "camera param: " << std::endl
                  << "fx: " << pinhole_model_->fx() << std::endl
                  << "fy: " << pinhole_model_->fy() << std::endl
                  << "cx: " << pinhole_model_->cx() << std::endl
                  << "cy: " << pinhole_model_->cy() << std::endl;
    }

    bool FeatureAlign::tracking(const std::vector<cv::Point2f> &vPointsRef, const std::vector<float> &vDepth, std::vector<cv::Point2f> &vPointsCur,
                                std::vector<uchar> &vMatches, Sophus::SE3 &transformation)
    {
        vDepthRef = vDepth;         //参考帧特征点深度
        vKeyPointsRef = vPointsRef; //参考帧特征点
        vKeyPointsCur = vPointsCur; //当前帧特征点
        vMatcheStates = vMatches;   //特征点匹配结果

        vbInliner_.resize(vPointsRef.size(), true);
        // TODO:使用金字塔优化

        optimize(transformation);

        std::cout << "feature based tracking results: " << std::endl
                  << transformation.matrix() << std::endl;

        return true;
    }

    void FeatureAlign::set_weightfunction()
    {
        if (tracker_info_.use_weight_scale)
        {
            switch (tracker_info_.scale_estimator_type)
            {
            case ScaleEstimatorType::MADScale:
                scale_estimator_.reset(new MADScaleEstimator());
                break;
            default:
                cerr << "Do not use scale estimator." << endl;
            }
        }

        switch (tracker_info_.weight_function_type)
        {
        case WeightFunctionType::TDistributionWeight:
            weight_function_.reset(new TDistributionWeightFunction());
            break;
        case WeightFunctionType::HuberWeight:
            weight_function_.reset(new HuberWeightFunction());
            break;
        default:
            cerr << "Do not use weight function." << endl;
        }
    }

    double FeatureAlign::compute_residuals(const Sophus::SE3 &Tcl)
    {
        errors_.clear();
        J_.clear();
        weight_.clear();

        n_measurement_ = 0;
        float chi2_raw = 0.0;
        for (size_t i = 0; i < vMatcheStates.size(); i++)
        {
            if (vMatcheStates[i] <= 0)
            {
                vbInliner_[i] = false;
                continue;
            }

            cv::Point2f uv_ref = vKeyPointsRef.at(i);
            cv::Point2f uv_cur = vKeyPointsCur.at(i);

            float depth = vDepthRef.at(i);
            if (depth <= 0)
            {
                vbInliner_[i] = false;
                continue;
            }

            Eigen::Vector3d pt_3d_ref = pinhole_model_->cam2world(Eigen::Vector2d(uv_ref.x, uv_ref.y)) * depth;
            Eigen::Vector3d pt_3d_cur = Tcl.rotation_matrix() * pt_3d_ref + Tcl.translation();
            Eigen::Vector2d uv_cur_proj = pinhole_model_->world2cam(pt_3d_cur);

            // 重投影误差
            Eigen::Vector2d err = Eigen::Vector2d(uv_cur.x, uv_cur.y) - uv_cur_proj;
            errors_.push_back(err.cast<float>());

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

            J_.push_back(J);
            weight_.push_back(1.0);

            chi2_raw += err.x() * err.x() + err.y() * err.y();
            n_measurement_++;
        }

        if (n_measurement_ > 0)
            return chi2_raw / n_measurement_;
        else
            return -1.0;
    }

    // implementation for LSQNonlinear class
    void FeatureAlign::update(const ModelType &old_model, ModelType &new_model)
    {
        Eigen::Matrix<double, 6, 1> update_;
        for (int i = 0; i < 6; i++)
            update_[i] = x_[i];
        new_model = Sophus::SE3::exp(update_) * old_model;
    }

    double FeatureAlign::build_LinearSystem(Sophus::SE3 &model)
    {
        double res = compute_residuals(model);

        H_.setZero();
        Jres_.setZero();

        if (tracker_info_.use_weight_scale)
        {
            // 使用权重函数
            std::vector<float> err_raw;
            for (int i = 0; i < errors_.size(); ++i)
            {
                Vector2 err_vec = errors_.at(i);
                err_raw.push_back(err_vec.x() * err_vec.x() + err_vec.y() * err_vec.y());
            }
            float mad_scale = scale_estimator_->compute(err_raw);

            if(verbose_)
                std::cout << "******** compute weights   **********, mad_scale: " << mad_scale << std::endl;

            // 计算权重
            for (size_t i = 0; i < errors_.size(); ++i)
            {
                float weight = weight_function_->weight(err_raw.at(i) / mad_scale);
                weight_.at(i) = weight;
            }
        }

        double res_weight = 0.0;
        for (int i = 0; i < errors_.size(); ++i)
        {
            Vector2 &res = errors_[i];
            Matrix2x6 &J = J_[i];
            float &weight = weight_[i];

            H_.noalias() += J.transpose() * J * weight;
            Jres_.noalias() -= J.transpose() * res * weight;

            res_weight += (res.x() * res.x() + res.y() * res.y()) * weight;
        }

        // return res_weight / n_measurement_;
        return res;
    }

}