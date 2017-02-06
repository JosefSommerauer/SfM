#ifndef GPU_ORB_H_INCLUDED
#define GPU_ORB_H_INCLUDED

#include "IFeatureDetect.h"
#include "GPUHelper.h"
#include "StaticAllocator.h"

#include <opencv2/xfeatures2d/cuda.hpp>
#include <cuda_runtime.h>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

class GPUOrb : public IFeatureDetect {

private:
  cv::Ptr<cv::cuda::ORB> f2d;
  StaticAllocator mImageAllocator;
  StaticAllocator mKeypointsAllocator;
  StaticAllocator mMaskAllocator;

  int mNorm;

public:
  GPUOrb(int nfeatures = 500);

  virtual bool findFeatures(Frame *frame);
  virtual ViewCombination matchFrames(Frame *left, Frame *right);
};

#endif
