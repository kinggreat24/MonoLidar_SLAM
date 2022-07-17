/*
 * @Author: Kinggreat24
 * @Date: 2020-05-19 22:25:01
 * @LastEditors: kinggreat24
 * @LastEditTime: 2022-07-12 13:38:31
 * @Description:  利用重投影误差计算相对运动
 */
#ifndef SPARSE_REPROJ_ALIGN_H
#define SPARSE_REPROJ_ALIGN_H

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

    class SparseReprojAlign : public LSQNonlinearGaussNewton<6, Sophus::SE3> //LSQNonlinearGaussNewton <6, Sophus::SE3f> LSQNonlinearLevenbergMarquardt <6, Sophus::SE3f>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SparseReprojAlign(const vk::PinholeCamera *pinhole_model, const ORB_SLAM2::_reproj_tracker_t &tracker_info);
        ~SparseReprojAlign();

        bool tracking(Frame *pReference, Frame *pCurrent, std::vector<int> mvMatches, Sophus::SE3 &transformation);

        virtual void startIteration();
        virtual void finishIteration();

    protected:
        // 计算雅克比矩阵与误差
        double compute_residuals(const Sophus::SE3 &transformation);

        // implementation for LSQNonlinear class
        virtual void update(const ModelType &old_model, ModelType &new_model);

    private:
        const vk::PinholeCamera *pinhole_model_;

        std::vector<Eigen::Vector3d> ref_keypoints_3d_; //参考帧特征点３Ｄ坐标
        std::vector<cv::KeyPoint> vKeyPointsRef;
        std::vector<cv::KeyPoint> vKeyPointsCur; //当前帧特征点
        std::vector<float> vDepthRef;
        std::vector<int> mvMatches_;
        std::vector<bool> vbInliner_; //是否为外点
        std::vector<float> pre_weight_res_p_;

        int NP;
        std::vector<float> errors_norm_; //重投影误差
        std::vector<Vector2> errors_;    //重投影误差
        std::vector<Matrix2x6> J_;       //雅克比矩阵
        std::vector<float> weight_;      //权重

        
    public:
        // weight function
        ORB_SLAM2::_reproj_tracker_t tracker_info_;
        bool use_weight_scale_;
        int max_level_;
        float scale_;
        std::shared_ptr<ScaleEstimator> scale_estimator_;
        std::shared_ptr<WeightFunction> weight_function_;
        void set_weightfunction();

    protected:
        virtual double build_LinearSystem(Sophus::SE3 &model);
    };

} // end of namespace dedvo

#endif //SPARSE_REPROJ_ALIGN_H