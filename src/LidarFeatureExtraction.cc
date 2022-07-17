/*
 * @Author: kinggreat24
 * @Date: 2022-05-24 17:25:04
 * @LastEditTime: 2022-05-24 21:29:22
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /ORB_SLAM2/src/LidarFeatureExtraction.cc
 * 可以输入预定的版权声明、个性签名、空行等
 */
#include "LidarFeatureExtraction.h"

namespace ORB_SLAM2
{
    LidarFeatureExtractor::LidarFeatureExtractor()
    {
        std::cout << "construct lidar feature extractor" << std::endl;
    }

    void LidarFeatureExtractor::FeatureExtraction(const std::string &lidar_files, lo::cloudblock_Ptr cblock)
    {
        // cblock = lo::cloudblock_Ptr(new lo::cloudblock_t());
        cblock->filename = lidar_files;
        dataio.read_pc_cloud_block(cblock, true);

        float vf_downsample_resolution_target = 0.0;
        float gf_grid_resolution = 2.0;
        float gf_max_grid_height_diff = 0.25;
        float gf_neighbor_height_diff = 1.2;
        float gf_max_height = DBL_MAX;
        int ground_down_rate = 10;
        int nonground_down_rate = 3;
        int dist_inv_sampling_method = 2;
        float dist_inv_sampling_dist = 15.0;
        bool pca_distance_adpative_on = 1.0;
        float pca_neigh_r = 1.0;
        int pca_neigh_k = 50;
        float pca_linearity_thre = 0.65;
        float pca_planarity_thre = 0.65;
        float pca_curvature_thre = 0.10;

        float pca_linearity_thre_down = pca_linearity_thre + 0.1;
        float pca_planarity_thre_down = pca_planarity_thre + 0.1;

        //Extract feature points
        cfilter.extract_semantic_pts(cblock, vf_downsample_resolution_target, gf_grid_resolution, gf_max_grid_height_diff,
                                     gf_neighbor_height_diff, gf_max_height, ground_down_rate, nonground_down_rate,
                                     pca_neigh_r, pca_neigh_k, pca_linearity_thre, pca_planarity_thre, pca_curvature_thre,
                                     pca_linearity_thre_down, pca_planarity_thre_down, pca_distance_adpative_on,
                                     dist_inv_sampling_method, dist_inv_sampling_dist);
    }
}
