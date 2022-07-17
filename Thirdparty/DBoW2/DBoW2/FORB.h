/*
 * @Author: kinggreat24
 * @Date: 2021-11-14 19:22:54
 * @LastEditTime: 2022-07-06 11:02:30
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /ORB_SLAM2/Thirdparty/DBoW2/DBoW2/FORB.h
 * 可以输入预定的版权声明、个性签名、空行等
 */
/**
 * File: FORB.h
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_F_ORB__
#define __D_T_F_ORB__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

#include "FClass.h"

namespace DBoW2 {

/// Functions to manipulate ORB descriptors
class FORB: protected FClass
{
public:

  /// Descriptor type
  typedef cv::Mat TDescriptor; // CV_8U
  /// Pointer to a single descriptor
  typedef const TDescriptor *pDescriptor;
  /// Descriptor length (in bytes)
  static const int L;

  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
  static void meanValue(const std::vector<pDescriptor> &descriptors,
    TDescriptor &mean);

  /**
   * Calculates the distance between two descriptors
   * @param a
   * @param b
   * @return distance
   */
  static int distance(const TDescriptor &a, const TDescriptor &b);

  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const TDescriptor &a);

  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(TDescriptor &a, const std::string &s);

  /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
  static void toMat32F(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat);

  static void toMat8U(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat);

  /**
   * Fills an descriptor with the values from an array
   * @param descriptors (out) descriptor
   * @param array (in) unsigned char * containing the values of the descriptor
   */
  static void fromArray8U(TDescriptor &descriptors, unsigned char * array);


};

} // namespace DBoW2

#endif

