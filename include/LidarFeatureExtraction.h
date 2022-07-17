/*
 * @Author: kinggreat24
 * @Date: 2022-05-24 17:22:16
 * @LastEditTime: 2022-05-26 11:02:20
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /ORB_SLAM2/include/LidarFeatureExtraction.h
 * 可以输入预定的版权声明、个性签名、空行等
 */
#ifndef LIDAR_FEATURE_EXTRACTION_H
#define LIDAR_FEATURE_EXTRACTION_H

#include "cfilter.hpp"
#include "dataio.hpp"
#include "cregistration.hpp"

namespace ORB_SLAM2
{
    class LidarFeatureExtractor
    {
    public:
        LidarFeatureExtractor();

        void FeatureExtraction(const std::string& lidar_files, lo::cloudblock_Ptr cblock);
    // private:
        lo::CFilter<Point_T> cfilter;
        lo::DataIo<Point_T> dataio;
        lo::CRegistration<Point_T> creg;
    };
} // namespace ORB_SLAM2

#endif //LIDAR_FEATURE_EXTRACTION_H