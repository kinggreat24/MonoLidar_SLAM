/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

#include <pcl-1.7/pcl/common/transforms.h>

namespace ORB_SLAM2
{

    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    std::ofstream Frame::times_ofs_;

    Frame::Frame()
    {
    }

    //Copy Constructor
    Frame::Frame(const Frame &frame)
        : mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
          mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
          mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
          mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
          mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
          mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
          mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
          mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
          mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
          mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
          mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
    {
        for (int i = 0; i < FRAME_GRID_COLS; i++)
            for (int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j] = frame.mGrid[i][j];

        if (!frame.mTcw.empty())
            SetPose(frame.mTcw);

        //
        // mpPointcloudCamera = frame.mpPointcloudCamera;
        mpMagLidarPointCloud = frame.mpMagLidarPointCloud;
        mvImgPyramid = frame.mvImgPyramid;
        mnTrackLevels = frame.mnTrackLevels;
        mpPinholeCamera = frame.mpPinholeCamera;
    }

    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
          mpReferenceKF(static_cast<KeyFrame *>(NULL))
    {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        // ORB extraction
        thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
        thread threadRight(&Frame::ExtractORB, this, 1, imRight);
        threadLeft.join();
        threadRight.join();

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ComputeStereoMatches();

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations)
        {
            ComputeImageBounds(imLeft);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
          mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
    {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        // ORB extraction
        ExtractORB(0, imGray);

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ComputeStereoFromRGBD(imDepth);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations)
        {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
          mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
    {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        // ORB extraction
        ExtractORB(0, imGray);

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        // Set no stereo information
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations)
        {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    //Monocular-Lidar
    Frame::Frame(const cv::Mat &imGray, const std::string &lidar_file, const double &timeStamp, LidarDepthExtration *pLidarDepthExtractor, ORBextractor *extractor, ORBVocabulary *voc, vk::PinholeCamera *pPinholeCamera, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
        : mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
          mpLidarDepthExtractor(pLidarDepthExtractor),
          mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
          mnTrackLevels(3), mpPinholeCamera(pPinholeCamera)
    {
        // Frame ID
        mnId = nNextId++;

        if (mnId == 0)
            times_ofs_ = std::ofstream("/home/kinggreat24/pc/times_dt_kitti.txt");

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        //读取激光点云数据
        pcl::PointCloud<pcl::PointXYZI>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZI>());
        ReadPointCloud(lidar_file, pCloud, true);

        // ORB extraction
        thread threadORB(&Frame::ExtractORB, this, 0, imGray);
        thread threadLidar(&Frame::LidarPreProcessing, this, pCloud, imGray);
        threadORB.join();
        threadLidar.join();

        N = mvKeys.size();
        if (mvKeys.empty())
            return;

        // 特征点去畸变
        UndistortKeyPoints();

        // 计算特征点深度信息
        ComputeStereoFromLidar(pCloud);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations)
        {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    void Frame::AssignFeaturesToGrid()
    {
        int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for (int i = 0; i < N; i++)
        {
            const cv::KeyPoint &kp = mvKeysUn[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractORB(int flag, const cv::Mat &im)
    {
        if (flag == 0)
            (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
        else
            (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
    }

    void Frame::SetPose(cv::Mat Tcw)
    {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::UpdatePoseMatrices()
    {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;
    }

    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
    {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        // Check positive depth
        if (PcZ < 0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = fx * PcX * invz + cx;
        const float v = fy * PcY * invz + cy;

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P - mOw;
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        pMP->mTrackProjXR = u - mbf * invz;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
    {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
        {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
            {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++)
                {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    if (bCheckLevels)
                    {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
    {
        posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
        posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
            return false;

        return true;
    }

    void Frame::ComputeBoW()
    {
        if (mBowVec.empty())
        {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void Frame::UndistortKeyPoints()
    {
        if (mDistCoef.at<float>(0) == 0.0)
        {
            mvKeysUn = mvKeys;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++)
        {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for (int i = 0; i < N; i++)
        {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUn[i] = kp;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft)
    {
        if (mDistCoef.at<float>(0) != 0.0)
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

    void Frame::ComputeStereoMatches()
    {
        mvuRight = vector<float>(N, -1.0f);
        mvDepth = vector<float>(N, -1.0f);

        const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

        const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

        //Assign keypoints to row table
        vector<vector<size_t>> vRowIndices(nRows, vector<size_t>());

        for (int i = 0; i < nRows; i++)
            vRowIndices[i].reserve(200);

        const int Nr = mvKeysRight.size();

        for (int iR = 0; iR < Nr; iR++)
        {
            const cv::KeyPoint &kp = mvKeysRight[iR];
            const float &kpY = kp.pt.y;
            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = ceil(kpY + r);
            const int minr = floor(kpY - r);

            for (int yi = minr; yi <= maxr; yi++)
                vRowIndices[yi].push_back(iR);
        }

        // Set limits for search
        const float minZ = mb;
        const float minD = 0;
        const float maxD = mbf / minZ;

        // For each left keypoint search a match in the right image
        vector<pair<int, int>> vDistIdx;
        vDistIdx.reserve(N);

        for (int iL = 0; iL < N; iL++)
        {
            const cv::KeyPoint &kpL = mvKeys[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;

            const vector<size_t> &vCandidates = vRowIndices[vL];

            if (vCandidates.empty())
                continue;

            const float minU = uL - maxD;
            const float maxU = uL - minD;

            if (maxU < 0)
                continue;

            int bestDist = ORBmatcher::TH_HIGH;
            size_t bestIdxR = 0;

            const cv::Mat &dL = mDescriptors.row(iL);

            // Compare descriptor to right keypoints
            for (size_t iC = 0; iC < vCandidates.size(); iC++)
            {
                const size_t iR = vCandidates[iC];
                const cv::KeyPoint &kpR = mvKeysRight[iR];

                if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                    continue;

                const float &uR = kpR.pt.x;

                if (uR >= minU && uR <= maxU)
                {
                    const cv::Mat &dR = mDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }

            // Subpixel match by correlation
            if (bestDist < thOrbDist)
            {
                // coordinates in image pyramid at keypoint scale
                const float uR0 = mvKeysRight[bestIdxR].pt.x;
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                const float scaleduL = round(kpL.pt.x * scaleFactor);
                const float scaledvL = round(kpL.pt.y * scaleFactor);
                const float scaleduR0 = round(uR0 * scaleFactor);

                // sliding window search
                const int w = 5;
                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);
                IL.convertTo(IL, CV_32F);
                IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

                int bestDist = INT_MAX;
                int bestincR = 0;
                const int L = 5;
                vector<float> vDists;
                vDists.resize(2 * L + 1);

                const float iniu = scaleduR0 + L - w;
                const float endu = scaleduR0 + L + w + 1;
                if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                    continue;

                for (int incR = -L; incR <= +L; incR++)
                {
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                    IR.convertTo(IR, CV_32F);
                    IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

                    float dist = cv::norm(IL, IR, cv::NORM_L1);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestincR = incR;
                    }

                    vDists[L + incR] = dist;
                }

                if (bestincR == -L || bestincR == L)
                    continue;

                // Sub-pixel match (Parabola fitting)
                const float dist1 = vDists[L + bestincR - 1];
                const float dist2 = vDists[L + bestincR];
                const float dist3 = vDists[L + bestincR + 1];

                const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

                if (deltaR < -1 || deltaR > 1)
                    continue;

                // Re-scaled coordinate
                float bestuR = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

                float disparity = (uL - bestuR);

                if (disparity >= minD && disparity < maxD)
                {
                    if (disparity <= 0)
                    {
                        disparity = 0.01;
                        bestuR = uL - 0.01;
                    }
                    mvDepth[iL] = mbf / disparity;
                    mvuRight[iL] = bestuR;
                    vDistIdx.push_back(pair<int, int>(bestDist, iL));
                }
            }
        }

        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        for (int i = vDistIdx.size() - 1; i >= 0; i--)
        {
            if (vDistIdx[i].first < thDist)
                break;
            else
            {
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
            }
        }
    }

    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
    {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++)
        {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            const float d = imDepth.at<float>(v, u);

            if (d > 0)
            {
                mvDepth[i] = d;
                mvuRight[i] = kpU.pt.x - mbf / d;
            }
        }
    }

    void Frame::ComputeStereoFromLidar(pcl::PointCloud<pcl::PointXYZI>::Ptr pCloud, const bool use_delanuy)
    {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        int depth_init_type = mpLidarDepthExtractor->depth_init_type_;
        if (DELAUNY_ALL == depth_init_type)
        {
            //将激光雷达转换到相机坐标系中
            if (mpPointcloudCamera == nullptr)
            {
                mpPointcloudCamera.reset(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(
                    *pCloud,
                    *mpPointcloudCamera,
                    mpLidarDepthExtractor->Tcam_lidar_);
            }

            // Get projected lidar points激光雷达已经转到相机坐标系中
            pcl::PointCloud<pcl::PointXYZI>::Ptr mpProjectPointCloud(new pcl::PointCloud<pcl::PointXYZI>());
            mpLidarDepthExtractor->pointCloudDepthFilter(mpProjectPointCloud, mpPointcloudCamera, "z", 5, 100);

            GEOM_FADE25D::Fade_2D *pdt_all = mpLidarDepthExtractor->CreateTerrain(mpProjectPointCloud);
            mpLidarDepthExtractor->PointFeatureDepthInit(pdt_all, mvKeysUn, mvDepth);

            for (size_t i = 0; i < mvDepth.size(); i++)
            {
                const cv::KeyPoint &kp = mvKeys[i];
                const cv::KeyPoint &kpU = mvKeysUn[i];

                float d = mvDepth[i];
                if (d > 0)
                {
                    mvuRight[i] = kpU.pt.x - mbf / d;
                }
            }
        }
        else if (DELAUNY_OBJ == depth_init_type)
        {
            // 基于目标三角网特征点深度拟合方法
            pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointCloud<pcl::PointXYZI>::Ptr obstacle_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());

            //(1)地面点云提取
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            double time_tokens = 0.0;
            mpLidarDepthExtractor->PatchWorkGroundSegmentation(pCloud, ground_cloud_ptr, obstacle_cloud_ptr, time_tokens);

            //(2)非地面点云分割
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            std::unordered_map<uint16_t, depth_clustering::Cloud> vLidarClusters;
            cv::Mat rangeSegImage;
            mpLidarDepthExtractor->LidarDepthClustering(obstacle_cloud_ptr, vLidarClusters, rangeSegImage);

            //(3)构造目标三角网
            std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
            GEOM_FADE25D::Fade_2D *pObj_delauny = mpLidarDepthExtractor->CreateTerrain(vLidarClusters);

            //(4)构造地面三角网
            //将激光雷达转换到相机坐标系中
            std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
            pcl::PointCloud<pcl::PointXYZI>::Ptr ground_pointcloud_camera_ptr(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::transformPointCloud(
                *ground_cloud_ptr,
                *ground_pointcloud_camera_ptr,
                mpLidarDepthExtractor->Tcam_lidar_);
            GEOM_FADE25D::Fade_2D *pGround_delauny = mpLidarDepthExtractor->CreateTerrain(ground_pointcloud_camera_ptr);

            // (5)基于目标的特征深度拟合
            std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
            mpLidarDepthExtractor->PointFeatureDepthInit(pObj_delauny, pGround_delauny, mvKeysUn, mvDepth);
            for (size_t i = 0; i < mvDepth.size(); i++)
            {
                const cv::KeyPoint &kp = mvKeys[i];
                const cv::KeyPoint &kpU = mvKeysUn[i];

                float d = mvDepth[i];
                if (d > 0)
                {
                    mvuRight[i] = kpU.pt.x - mbf / d;
                }
            }

            std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();

            double t_duration1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
            double t_duration2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
            double t_duration3 = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count();
            double t_duration4 = std::chrono::duration_cast<std::chrono::duration<double>>(t5 - t4).count();
            double t_duration5 = std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t5).count();

            double t_all = std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t1).count();

            std::cout << "ground segmentation time: " << t_duration1
                      << "  lidar clustering time: " << t_duration2
                      << "  obj delaunay time: " << t_duration3
                      << "  ground delaunay time: " << t_duration4
                      << " feature depth init: " << t_duration5 << std::endl;
            std::cout << "All time: " << t_all << std::endl;

            // if (mnId == 0)
            // {
            //     std::ofstream times_ofs("/home/kinggreat24/times.txt", w);

            // }

            times_ofs_ << setprecision(9) << t_duration1 << " " << t_duration2 << " " << t_duration3 + t_duration4 << " " << t_duration5 << std::endl;
            times_ofs_.flush();

            // reset pointcloud
            ground_cloud_ptr.reset(new pcl::PointCloud<pcl::PointXYZI>());
            obstacle_cloud_ptr.reset(new pcl::PointCloud<pcl::PointXYZI>());
            ground_pointcloud_camera_ptr.reset(new pcl::PointCloud<pcl::PointXYZI>());
            pCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        }
        else if (LIMO == depth_init_type)
        {
            //使用Limo的方法进行特征深度拟合
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            Eigen::VectorXd feature_depths;
            mpLidarDepthExtractor->CalculateFeatureDepthsCurFrame(pCloud, mvKeysUn, feature_depths);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double t_duration1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
            std::cout << "limo depth extraction time: " << t_duration1 << std::endl;

            for (size_t i = 0; i < feature_depths.size(); i++)
            {
                const cv::KeyPoint &kp = mvKeys[i];
                const cv::KeyPoint &kpU = mvKeysUn[i];

                float d = feature_depths[i];
                if (d > 0)
                {
                    mvDepth[i] = d;
                    mvuRight[i] = kpU.pt.x - mbf / d;
                }
            }

            times_ofs_ << setprecision(9) << t_duration1 << std::endl;
            times_ofs_.flush();
        }
        else if (CAMVOX == depth_init_type)
        {
            // 使用camvox的方式提取特征点深度
            //将激光雷达转换到相机坐标系中
            // pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_camera(new pcl::PointCloud<pcl::PointXYZI>());
            // pcl::transformPointCloud(
            //     *pCloud,
            //     *pointcloud_camera,
            //     mpLidarDepthExtractor->Tcam_lidar_);

            if (mpPointcloudCamera == nullptr)
            {
                mpPointcloudCamera.reset(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(
                    *pCloud,
                    *mpPointcloudCamera,
                    mpLidarDepthExtractor->Tcam_lidar_);
            }

            // Get projected lidar points激光雷达已经转到相机坐标系中
            pcl::PointCloud<pcl::PointXYZI>::Ptr mpProjectPointCloud(new pcl::PointCloud<pcl::PointXYZI>());
            mpLidarDepthExtractor->pointCloudDepthFilter(mpProjectPointCloud, mpPointcloudCamera, "z", 5, 100);
            mpLidarDepthExtractor->CalculateFeatureDepthsCamVox(mpProjectPointCloud, mvKeysUn, mvDepth);

            for (size_t i = 0; i < mvDepth.size(); i++)
            {
                float d = mvDepth.at(i);
                if (d > 0)
                {
                    const cv::KeyPoint &kpU = mvKeysUn[i];
                    mvuRight[i] = kpU.pt.x - mbf / d;
                }
            }
        }
    }

    cv::Mat Frame::UnprojectStereo(const int &i)
    {
        const float z = mvDepth[i];
        if (z > 0)
        {
            const float u = mvKeysUn[i].pt.x;
            const float v = mvKeysUn[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
            return mRwc * x3Dc + mOw;
        }
        else
            return cv::Mat();
    }

    int Frame::ReadPointCloud(const std::string &file, pcl::PointCloud<pcl::PointXYZI>::Ptr outpointcloud, bool isBinary)
    {
        // pcl::PointCloud<PointType>::Ptr curPointCloud(new pcl::PointCloud<PointType>());
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
                float dist = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
                if (dist < 2)
                    continue;

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

        // SavePointCloudPly("/home/bingo/pc/loam/pointcloud/pc.ply",outpointcloud);
        return outpointcloud->points.size();
    }

    void Frame::DrawFeatures(const cv::Mat &imGray)
    {
        cv::Mat imOut = imGray.clone();
        if (imOut.channels() == 1)
            cv::cvtColor(imOut, imOut, CV_GRAY2BGR);

        const float r = 5.0;
        float v_min = 5.0;
        float v_max = 80.0;
        float dv = v_max - v_min;
        for (size_t i = 0; i < mvKeys.size(); i++)
        {
            cv::Point2f pt1, pt2;
            pt1.x = mvKeys[i].pt.x - r;
            pt1.y = mvKeys[i].pt.y - r;
            pt2.x = mvKeys[i].pt.x + r;
            pt2.y = mvKeys[i].pt.y + r;

            float v = mvDepth[i];
            if (v <= 0)
                continue;

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

            cv::rectangle(imOut, pt1, pt2, 255 * cv::Scalar(r, g, b));
            cv::circle(imOut, mvKeys[i].pt, 2, 255 * cv::Scalar(r, g, b), -1);
        }

        // cv::imwrite("mono_lidar_depth.png", imOut);
    }

    // 激光点预处理
    void Frame::LidarPreProcessing(const pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudLidar, const cv::Mat &imGray)
    {
        // 图像金字塔生成
        PyrDownImage(imGray);

        // 将激光点投影到相机坐标系中
        mpPointcloudCamera.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(
            *pCloudLidar,
            *mpPointcloudCamera,
            mpLidarDepthExtractor->Tcam_lidar_);

        // 过滤
        pcl::PointCloud<pcl::PointXYZI>::Ptr mpProjectPointCloud(new pcl::PointCloud<pcl::PointXYZI>());
        mpLidarDepthExtractor->pointCloudDepthFilter(mpProjectPointCloud, mpPointcloudCamera, "z", 3, 100);

        // 激光点采样
        LidarPointsSampling(mvImgPyramid[0], *mpProjectPointCloud);
    }

    template <typename T>
    void Frame::PyrDownMeanSmooth(const cv::Mat &in, cv::Mat &out)
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

    // 激光点采样
    void Frame::LidarPointsSampling(const cv::Mat &original_img, const pcl::PointCloud<pcl::PointXYZI> &pc_camera)
    {
        //激光点投影到图像
        int num_bucket_size = 20;
        std::vector<std::pair<float, pcl::PointXYZI>> mag_point_bucket;
        mag_point_bucket.reserve(num_bucket_size);
        mpMagLidarPointCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());

        for (auto pt = pc_camera.begin(); pt != pc_camera.end(); pt++)
        {
            Eigen::Vector3d xyz(pt->x, pt->y, pt->z);
            if (pt->z < 0)
                continue;

            Eigen::Vector2d uv = mpPinholeCamera->world2cam(xyz);
            int u = static_cast<int>(uv(0));
            int v = static_cast<int>(uv(1));

            if (mpPinholeCamera->isInFrame(Eigen::Vector2i(u, v), 4))
            {
                float dx = 0.5f * (original_img.at<float>(v, u + 1) - original_img.at<float>(v, u - 1));
                float dy = 0.5f * (original_img.at<float>(v + 1, u) - original_img.at<float>(v - 1, u));

                std::pair<float, pcl::PointXYZI> mag_point;
                mag_point = make_pair((dx * dx + dy * dy), (*pt));

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

    void Frame::PyrDownImage(const cv::Mat &imGray)
    {
        //图像金字塔
        cv::Mat original_img_ = imGray.clone();
        original_img_.convertTo(original_img_, CV_32FC1, 1.0 / 255);

        mvImgPyramid.resize(mnTrackLevels);
        mvImgPyramid[0] = original_img_;
        for (int i = 1; i < mnTrackLevels; i++)
        {
            mvImgPyramid[i] = cv::Mat(cv::Size(mvImgPyramid[i - 1].size().width / 2, mvImgPyramid[i - 1].size().height / 2), mvImgPyramid[i - 1].type());
            PyrDownMeanSmooth<float>(mvImgPyramid[i - 1], mvImgPyramid[i]);
        }
    }

} //namespace ORB_SLAM
