#include "Camera.h"
#include <iostream>
#include <cassert>

Camera::Camera(size_t const w, size_t const h, cv::Mat const & K, cv::Mat const & kappa) : mWidth(w), mHeight(h)
{
  mK = K;
  mKappa = kappa;

  assert(mK.cols == 3);
  assert(mK.rows == 3);
  assert(mKappa.cols == 5);
  assert(mKappa.rows == 1);
}

Camera::Camera(size_t w, size_t h)  : mWidth(w), mHeight(h) {
  cv::Mat K = (cv::Mat_<double>(3, 3) << 922.45051,   0.00147, 635.39454,
										   0      , 920.41192, 354.20846,
										   0      ,   0      ,   1);
  cv::Mat kappa = (cv::Mat_<double>(1, 5) << 0.10620, -0.20464, 0, 0, 0);

  mK = K;
  mKappa = kappa;

  std::cout << "Actul Camera Resolution: " << mWidth << "*" << mHeight << std::endl << std::endl;

  assert(mK.cols == 3);
  assert(mK.rows == 3);
  assert(mKappa.cols == 5);
  assert(mKappa.rows == 1);
}

cv::Mat const &Camera::getCalibrationMatrix() const
{
  return mK;
}

void Camera::setCalibrationMatrix(cv::Mat K) {
	mK = K;

	assert(mK.cols == 3);
	assert(mK.rows == 3);
}

cv::Mat const &Camera::getDistortionCoefficients() const
{
  return mKappa;
}

size_t Camera::getWidth() const
{
  return mWidth;
}

size_t Camera::getHeight() const
{
  return mHeight;
}


