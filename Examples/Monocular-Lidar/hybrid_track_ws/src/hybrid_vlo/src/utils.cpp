/*
 * @Author: your name
 * @Date: 2021-09-20 15:10:25
 * @LastEditTime: 2021-09-29 13:56:27
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /NLS_examples/src/utils.cpp
 */

#include "utils.h"

#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/io/pcd_io.h>

using namespace std;

namespace hybrid_vlo
{
    void LoadGroundTruth(const std::string &gt_file, std::vector<Eigen::Matrix<double, 4, 4>> &gtPoses)
    {
        FILE *fp = fopen(gt_file.c_str(), "r");
        if (!fp)
            return;

        while (!feof(fp))
        {
            Eigen::Matrix<double, 3, 4> P;
            if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                       &P(0, 0), &P(0, 1), &P(0, 2), &P(0, 3),
                       &P(1, 0), &P(1, 1), &P(1, 2), &P(1, 3),
                       &P(2, 0), &P(2, 1), &P(2, 2), &P(2, 3)) == 12)
            {
                Eigen::Matrix4d gt_pose = Eigen::Matrix4d::Identity();
                gt_pose.block<3, 4>(0, 0) = P;
                gtPoses.push_back(gt_pose);
            }
        }
        fclose(fp);
    }

    // 读取数据
    void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                    vector<string> &vstrImageRight, vector<string> &vstrImageDisparity, vector<string> &vstrLidar, vector<double> &vTimestamps)
    {
        ifstream fTimes;
        string strPathTimeFile = strPathToSequence + "/times.txt";
        fTimes.open(strPathTimeFile.c_str());
        while (!fTimes.eof())
        {
            string s;
            getline(fTimes, s);
            if (!s.empty())
            {
                stringstream ss;
                ss << s;
                double t;
                ss >> t;
                vTimestamps.push_back(t);
            }
        }

        string strPrefixLeft = strPathToSequence + "/image_0/";
        string strPrefixRight = strPathToSequence + "/image_1/";
        string strPrefixLidar = strPathToSequence + "/velodyne/";
        string strPrefixDisparity = strPathToSequence + "/disparity/";

        const int nTimes = vTimestamps.size();
        vstrImageLeft.resize(nTimes);
        vstrImageRight.resize(nTimes);
        vstrLidar.resize(nTimes);
        vstrImageDisparity.resize(nTimes);

        for (int i = 0; i < nTimes; i++)
        {
            stringstream ss;
            ss << setfill('0') << setw(6) << i;
            vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
            vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
            vstrImageDisparity[i] =  strPrefixDisparity + ss.str() + ".png";
            vstrLidar[i] = strPrefixLidar + ss.str() + ".bin";
        }
    }

    int ReadPointCloud(const std::string &file, pcl::PointCloud<pcl::PointXYZI>::Ptr outpointcloud, bool isBinary)
    {
        if (isBinary)
        {
            // load point cloud
            std::fstream input(file.c_str(), std::ios::in | std::ios::binary);
            if (!input.good())
            {
                std::cerr << "Could not read file: " << file << std::endl;
                exit(EXIT_FAILURE);
            }
            //LOG(INFO)<<"Read: "<<file<<std::endl;

            for (int i = 0; input.good() && !input.eof(); i++)
            {
                pcl::PointXYZI point;
                input.read((char *)&point.x, 3 * sizeof(float));
                input.read((char *)&point.intensity, sizeof(float));

                //remove all points behind image plane (approximation)
                /*if (point.x < mMinDepth)
                continue;*/
                outpointcloud->points.push_back(point);
            }
        }
        else
        {
            if (-1 == pcl::io::loadPCDFile<pcl::PointXYZI>(file, *outpointcloud))
            {
                std::cerr << "Could not read file: " << file << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        outpointcloud->height = 1;
        outpointcloud->width = outpointcloud->points.size();

        return outpointcloud->points.size();
    }

    template <typename T>
    void pyrDownMeanSmooth(const cv::Mat &in, cv::Mat &out)
    {
        out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

        // #pragma omp parallel for collapse(2)
        for (int y = 0; y < out.rows; ++y)
        {
            for (int x = 0; x < out.cols; ++x)
            {
                int x0 = x * 2;
                int x1 = x0 + 1;
                int y0 = y * 2;
                int y1 = y0 + 1;

                out.at<T>(y, x) = (T)((in.at<T>(y0, x0) + in.at<T>(y0, x1) + in.at<T>(y1, x0) + in.at<T>(y1, x1)) / 4.0f);
            }
        }
    }

    void create_image_pyramid(const cv::Mat &img_level_0, int n_levels, std::vector<cv::Mat> &pyramid)
    {
        pyramid.resize(n_levels);
        pyramid[0] = img_level_0;

        for (int i = 1; i < n_levels; ++i)
        {
            pyramid[i] = cv::Mat(pyramid[i - 1].rows / 2, pyramid[i - 1].cols / 2, CV_32FC1);
            pyrDownMeanSmooth<float>(pyramid[i - 1], pyramid[i]);
        }
    }

    // 显示函数
    void ShowPointClouds(const pcl::PointCloud<pcl::PointXYZI>::Ptr &mpLidarPointCloud, const std::vector<cv::Mat> &mvImgPyramid,
                         const vk::PinholeCamera *mpPinholeCamera, cv::Mat &image_out, size_t num_level)
    {
        cv::Mat img = mvImgPyramid[num_level];
        if (img.channels() == 1)
        {
            cv::cvtColor(img, image_out, cv::COLOR_GRAY2BGR);
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
            uv_ref = mpPinholeCamera->world2cam(xyz_ref) * scale;

            if (!mpPinholeCamera->isInFrame(uv_ref.cast<int>(), 0))
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

            cv::circle(image_out, cv::Point(u_ref_i, v_ref_i), 2.0, cv::Scalar(r, g, b), -1);
        }
        image_out.convertTo(image_out, CV_8UC3, 255);
    }

    Eigen::Vector3d get_false_color(float v, const float v_min, const float v_max)
    {
        float dv = v_max - v_min;
        float r = 1.0, g = 1.0, b = 1.0;
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

        return Eigen::Vector3d(r, g, b);
    }

    int klt_feature_tracking(const cv::Mat &imgrayLast, const cv::Mat &imgrayCur, const std::vector<cv::KeyPoint> &last_keypoints, std::vector<cv::Point2f> &prevpoints, std::vector<cv::Point2f> &nextpoints,
                             std::vector<uchar> &tracked_states, const cv::Mat &dynamic_mask, cv::Mat &debug_image, bool flow_back)
    {
        // Detect dynamic target and ultimately optput the T matrix
        // std::vector<cv::Point2f> prepoint, nextpoint;
        // std::vector<uchar> forward_state;
        std::vector<float> forward_err;

        for (size_t i = 0; i < last_keypoints.size(); i++)
        {
            if (dynamic_mask.empty())
                prevpoints.push_back(last_keypoints.at(i).pt);
        }

        // cv::goodFeaturesToTrack(imgrayLast, prepoint, 1000, 0.01, 8, dynamic_mask, 3, true, 0.04);
        // cv::cornerSubPix(imgrayLast, prepoint, cv::Size(10, 10), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
        // 正向光流
        // cv::calcOpticalFlowPyrLK(imgrayLast, imgrayCur, prepoint, nextpoint, forward_state, forward_err, cv::Size(22, 22), 5, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01),cv::OPTFLOW_USE_INITIAL_FLOW);
        cv::calcOpticalFlowPyrLK(imgrayLast, imgrayCur, prevpoints, nextpoints, tracked_states, forward_err, cv::Size(22, 22), 5,
                                 cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));

        std::cout << "ref kp size: " << prevpoints.size() << " tracked_states size: " << tracked_states.size() << " nextpoint size: " << nextpoints.size() << std::endl;

        const float limit_edge_corner = 5.0;
        const float limit_dis_epi = 1.0;
        const float limit_of_check = 2210;

        int inlier_points = 0;
        // reverse check
        if (flow_back)
        {
            std::vector<uchar> reverse_state(tracked_states.size());
            std::vector<float> reverse_err;
            std::vector<cv::Point2f> reverse_pts, reverse_nextpoint;
            reverse_pts = nextpoints;
            cv::calcOpticalFlowPyrLK(imgrayCur, imgrayLast, reverse_pts, reverse_nextpoint, reverse_state, reverse_err, cv::Size(22, 22), 5, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));

            for (int i = 0; i < tracked_states.size(); i++)
            {
                cv::Point2f dif_pt = reverse_nextpoint[i] - prevpoints[i];
                float dist = std::sqrt(dif_pt.x * dif_pt.x + dif_pt.y * dif_pt.y);
                if (tracked_states[i] == 0 || reverse_state[i] == 0 || dist > limit_dis_epi)
                {
                    tracked_states[i] = 0;
                    continue;
                }

                int dx[10] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
                int dy[10] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
                int x1 = prevpoints[i].x, y1 = prevpoints[i].y;
                int x2 = nextpoints[i].x, y2 = nextpoints[i].y;
                if ((x1 < limit_edge_corner || x1 >= imgrayLast.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= imgrayLast.cols - limit_edge_corner ||
                     y1 < limit_edge_corner || y1 >= imgrayLast.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= imgrayLast.rows - limit_edge_corner))
                {
                    tracked_states[i] = 0;
                    continue;
                }

                //统计像素光度误差
                double sum_check = 0;
                for (int j = 0; j < 9; j++)
                    sum_check += abs(imgrayLast.at<uchar>(y1 + dy[j], x1 + dx[j]) - imgrayCur.at<uchar>(y2 + dy[j], x2 + dx[j]));

                if (sum_check > limit_of_check)
                {
                    tracked_states[i] = 0;
                    continue;
                }

                inlier_points++;
                tracked_states[i] = 1;
            }
        }
        else
        {
            for (int i = 0; i < tracked_states.size(); i++)
            {
                if (tracked_states[i] != 0)
                {
                    int dx[10] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
                    int dy[10] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
                    int x1 = prevpoints[i].x, y1 = prevpoints[i].y;
                    int x2 = nextpoints[i].x, y2 = nextpoints[i].y;
                    if ((x1 < limit_edge_corner || x1 >= imgrayLast.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= imgrayLast.cols - limit_edge_corner || y1 < limit_edge_corner || y1 >= imgrayLast.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= imgrayLast.rows - limit_edge_corner))
                    {
                        tracked_states[i] = 0;
                        continue;
                    }
                    //统计像素误差
                    double sum_check = 0;
                    for (int j = 0; j < 9; j++)
                        sum_check += abs(imgrayLast.at<uchar>(y1 + dy[j], x1 + dx[j]) - imgrayCur.at<uchar>(y2 + dy[j], x2 + dx[j]));

                    if (sum_check > limit_of_check)
                        tracked_states[i] = 0;

                    if (tracked_states[i])
                    {
                        inlier_points++;
                    }
                }
            }
        }

        // tracked_states = forward_state;

        return inlier_points;
    }

    cv::Mat showTrackingResults(const cv::Mat& imgrayLast, const cv::Mat& imgrayCur, const std::vector<cv::Point2f> &prevpoints, std::vector<cv::Point2f> &nextpoints, std::vector<uchar> &tracked_states)
    {
        //显示光流结果
        cv::Mat tracking_image;
        int nCols = imgrayLast.cols;
        int nRows = imgrayLast.rows;
        cv::vconcat(imgrayLast, imgrayCur, tracking_image);
        if (tracking_image.channels() == 1)
            cv::cvtColor(tracking_image, tracking_image, CV_GRAY2BGR);

        for (size_t i = 0; i < prevpoints.size(); i++)
        {
            if(tracked_states[i] <= 0)
                continue;
            cv::circle(tracking_image, prevpoints.at(i), 2.0, cv::Scalar(0, 255, 0));
            cv::circle(tracking_image, nextpoints.at(i) + cv::Point2f(0, nRows), 2.0, cv::Scalar(255, 0, 0));
            cv::line(tracking_image, prevpoints.at(i), nextpoints.at(i) + cv::Point2f(0, nRows), cv::Scalar(0, 0, 255), 1, CV_AA);
        }
        return tracking_image.clone();
    }

} //end of namespace