/*
 * @Author: kinggreat24
 * @Date: 2022-07-08 00:25:59
 * @LastEditTime: 2022-07-12 14:07:37
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /ORB_SLAM2/include/Common.h
 * 可以输入预定的版权声明、个性签名、空行等
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <vector>

#include "lidar_sparse_align/WeightFunction.h"

namespace ORB_SLAM2
{
    typedef Eigen::Matrix<float, 2, 1> Vector2;
    typedef Eigen::Matrix<double, 4, 1> Vector4d;
    typedef Eigen::Matrix<float, 6, 1> Vector6;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<float, 2, 6> Matrix2x6;
    typedef Eigen::Matrix<double, 4, 4> Matrix4d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    // Input sensor
    enum eDepthInitMethod
    {
        DELAUNY_ALL = 0,
        DELAUNY_OBJ = 1,
        LIMO = 2,
        CAMVOX = 3
    };

    // Input sensor
    enum eLidarSensorType
    {
        VLP_16 = 0,
        HDL_32 = 1,
        HDL_64 = 2,
        HDL_64_EQUAL = 3,
        RFANS_16 = 4,
        CFANS_32 = 5
    };

    //地面点云提取参数
    typedef struct _patchwork_param_t patchwork_param_t;
    struct _patchwork_param_t
    {
        double sensor_height_;
        bool verbose_;

        int num_iter_;
        int num_lpr_;
        int num_min_pts_;
        double th_seeds_;
        double th_dist_;
        double max_range_;
        double min_range_;
        int num_rings_;
        int num_sectors_;
        double uprightness_thr_;
        double adaptive_seed_selection_margin_;

        // For global threshold
        bool using_global_thr_;
        double global_elevation_thr_;

        int num_zones_;
        std::vector<int> num_sectors_each_zone_;
        std::vector<int> num_rings_each_zone_;
        std::vector<double> min_ranges_;
        std::vector<double> elevation_thr_;
        std::vector<double> flatness_thr_;
    };

    // 直接法跟踪参数
    typedef struct _tracker_t tracker_t;
    struct _tracker_t
    {
        int levels;
        int min_level;
        int max_level;
        int max_iteration;

        bool use_weight_scale = true;
        string scale_estimator;
        string weight_function;

        ScaleEstimatorType scale_estimator_type;
        WeightFunctionType weight_function_type;

        void set_scale_estimator_type()
        {
            if (!scale_estimator.compare("None"))
                use_weight_scale = false;
            if (!scale_estimator.compare("TDistributionScale"))
                scale_estimator_type = ScaleEstimatorType::TDistributionScale;

            cerr << "ScaleType : " << static_cast<int>(scale_estimator_type) << std::endl;
        }

        void set_weight_function_type()
        {
            if (!weight_function.compare("TDistributionWeight"))
                weight_function_type = WeightFunctionType::TDistributionWeight;
        }
    };

    // 间接法相关参数
    typedef struct _reproj_tracker_t reproj_tracker_t;
    struct _reproj_tracker_t
    {
        int levels;
        int max_iteration;

        bool use_weight_scale = true;
        string scale_estimator;
        string weight_function;

        ScaleEstimatorType scale_estimator_type;
        WeightFunctionType weight_function_type;

        void set_scale_estimator_type()
        {
            if (!scale_estimator.compare("None"))
                use_weight_scale = false;
            if (!scale_estimator.compare("MadScale"))
                scale_estimator_type = ScaleEstimatorType::MADScale;

            cerr << "reproj_tracker_t ScaleType : " << static_cast<int>(scale_estimator_type) << std::endl;
        }

        void set_weight_function_type()
        {
            if (!weight_function.compare("HuberWeight"))
                weight_function_type = WeightFunctionType::HuberWeight;
        }
    };
}

#endif //