/*
 * @Author: your name
 * @Date: 2021-09-21 22:29:12
 * @LastEditTime: 2021-09-23 15:48:58
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /NLS_examples/include/lidar_sparse_align/feature_align.h
 */
#ifndef FEATURE_ALIGN_H
#define FEATURE_ALIGN_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <sophus/se3.h>

#include "lidar_sparse_align/WeightFunction.h"
#include "lidar_sparse_align/LSQNonlinear.hpp"

#include "Frame.h"

#include <vikit/pinhole_camera.h>
#include <vikit/vision.h>

namespace hybrid_vlo
{

    // 间接法
    class FeatureAlign : public LSQNonlinearGaussNewton<6, Sophus::SE3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        FeatureAlign(const vk::PinholeCamera *pinhole_model, const _tracker_t &tracker_info);
        ~FeatureAlign() {}

        bool tracking(const std::vector<cv::Point2f> &vPointsRef, const std::vector<float>& vDepth, std::vector<cv::Point2f> &vPointsCur,
                      std::vector<uchar> &vMatches, Sophus::SE3 &transformation);

    protected:
        double compute_residuals(const Sophus::SE3 &transformation);
        void set_weightfunction();

        virtual double build_LinearSystem(Sophus::SE3 &model);
        // implementation for LSQNonlinear class
        virtual void update(const ModelType &old_model, ModelType &new_model);

    private:
        const vk::PinholeCamera *pinhole_model_;
        _tracker_t tracker_info_;

        std::vector<float> vDepthRef;           //参考帧特征点深度
        std::vector<cv::Point2f> vKeyPointsRef; //参考帧特征点
        std::vector<cv::Point2f> vKeyPointsCur; //当前帧特征点
        std::vector<uchar> vMatcheStates;       //特征点匹配结果
        std::vector<bool> vbInliner_;           //是否是内点

        std::vector<Vector2> errors_; //重投影误差
        std::vector<Matrix2x6> J_;    //雅克比矩阵
        std::vector<float> weight_;   //权重




        // 权重函数
        std::shared_ptr<ScaleEstimator> scale_estimator_;
        std::shared_ptr<WeightFunction> weight_function_;
    };

} // end of namespace hybrid_vlo

#endif //FEATURE_ALIGN_H