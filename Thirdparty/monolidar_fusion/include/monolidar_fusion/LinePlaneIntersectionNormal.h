/*
 * LinePlaneIntersectionNormal.h
 *
 *  Created on: Mar 15, 2017
 *      Author: wilczynski
 */

#pragma once

#include "LinePlaneIntersectionBase.h"

namespace Mono_Lidar {
class LinePlaneIntersectionNormal : public LinePlaneIntersectionBase {
public:
    LinePlaneIntersectionNormal() {
    }

    bool GetIntersection(const Eigen::Hyperplane<double, 3>& plane,
                         const Eigen::Vector3d& n0,
                         const Eigen::Vector3d& n1,
                         Eigen::Vector3d& intersectionPoint,
                         double& intersectionDistance) override;
};
}
