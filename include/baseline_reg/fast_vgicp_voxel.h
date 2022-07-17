//Copyright (c) 2019, k.koide

#ifndef FAST_GICP_FAST_VGICP_VOXEL_H
#define FAST_GICP_FAST_VGICP_VOXEL_H

namespace koide_reg {

class Vector3iHash {
public:
  size_t operator()(const Eigen::Vector3i& x) const {
    size_t seed = 0;
    boost::hash_combine(seed, x[0]);
    boost::hash_combine(seed, x[1]);
    boost::hash_combine(seed, x[2]);
    return seed;
  }
};

struct GaussianVoxel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<GaussianVoxel>;

  GaussianVoxel() {
    num_points = 0;
    mean.setZero();
    cov.setZero();
  }
  virtual ~GaussianVoxel() {}

  virtual void append(const Eigen::Vector4f& mean_, const Eigen::Matrix4f& cov_) = 0;

  virtual void finalize() = 0;

public:
  int num_points;
  Eigen::Vector4f mean;
  Eigen::Matrix4f cov;
};

struct MultiplicativeGaussianVoxel : GaussianVoxel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MultiplicativeGaussianVoxel() : GaussianVoxel() {}
  virtual ~MultiplicativeGaussianVoxel() {}

  virtual void append(const Eigen::Vector4f& mean_, const Eigen::Matrix4f& cov_) override {
    num_points++;
    Eigen::Matrix4f cov_inv = cov_;
    cov_inv(3, 3) = 1;
    cov_inv = cov_inv.inverse().eval();

    cov += cov_inv;
    mean += cov_inv * mean_;
  }

  virtual void finalize() override {
    cov(3, 3) = 1;
    mean[3] = 1;

    cov = cov.inverse().eval();
    mean = (cov * mean).eval();
  }
};

struct AdditiveGaussianVoxel : GaussianVoxel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AdditiveGaussianVoxel() : GaussianVoxel() {}
  virtual ~AdditiveGaussianVoxel() {}

  virtual void append(const Eigen::Vector4f& mean_, const Eigen::Matrix4f& cov_) override {
    num_points++;
    mean += mean_;
    cov += cov_;
  }

  virtual void finalize() override {
    mean /= num_points;
    cov /= num_points;
  }
};

}  // namespace koide_reg

#endif