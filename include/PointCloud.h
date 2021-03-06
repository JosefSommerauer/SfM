#ifndef POINTCLOUD_H_INCLUDED
#define POINTCLOUD_H_INCLUDED

#include <map>
#include <opencv2/core.hpp>
#include <opencv2/viz/types.hpp>
#include <pcl/common/common.h>

#include <iostream>
#include <string>

#include "Frame.h"

class PointCloud {

private:
  const double REPROJECTION_ERROR_THRESHOLD = 100;
  std::vector<cv::Vec4d> mPoints;
  std::vector<double> mReprojectionErrors;

  // map containing an entry for each frame that contributed to the pointcloud
  // each entry is associated with a vector of 2dpt_idx - 3dpt_idx pair
  std::map<Frame const*, std::vector<std::pair<size_t, size_t>>> mCorresponding2D;

  // vector for each 3D point -> for each frame correspond idx of imgpt for 3D point
  std::vector< std::vector<int> > mCorresponding3D;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr createPclPointCloud();

  //void keyboardcb(const cv::viz::KeyboardEvent& ke );

public:
  PointCloud() {}

  void addPoint(cv::Vec4d point, double error,
                std::pair<Frame const *, size_t> left_imgpt,
                std::pair<Frame const *, size_t> right_imgpt);

  void removePoint(size_t const pt3D_idx);

  cv::Vec4d const &getPoint(int i) const;
  double const &getError(int i) const;

  int findCorresponding3D(Frame const *frame, size_t const imgpt_idx) const;

  cv::Vec3b getColorOf3Dform2DPoint(size_t const pt3D_idx) const;
  cv::Vec3b getColorErrorRate(size_t const pt3D_idx) const;

  double getAverageError() const;

  cv::Point3d getPoint3D(size_t i) const;
  std::vector<int> getImgptForImg(size_t const pt3D_idx) const;

  std::vector<cv::Point3d> getPoints3D() const;
  std::vector<cv::Vec4d> getPoints() const;
  size_t size() const;
  void clearPoints();
  bool isViewable(size_t i) const;

  void RunVisualization(std::string name = "SfM");

  bool UpdatePoints(std::vector<cv::Vec4d> newPtCl);
};

#endif
