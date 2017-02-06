#ifndef GPU_SURF_H_INCLUDED
#define GPU_SURF_H_INCLUDED

#include "IFeatureDetect.h"
#include "GPUHelper.h"
#include "StaticAllocator.h"

#include <opencv2/xfeatures2d/cuda.hpp>
#include <cuda_runtime.h>

class GPUSurf : public IFeatureDetect {

private:
  cv::cuda::SURF_CUDA mSurf;
  StaticAllocator mImageAllocator;
  StaticAllocator mKeypointsAllocator;
  StaticAllocator mMaskAllocator;

  int mNorm;

public:
  GPUSurf(double hessian = 500, int octaves = 5, int layers = 5,
          bool extended = true, bool upright = true,
          int norm = cv::NORM_L2, float ratio = 0.1);

  virtual bool findFeatures(Frame *frame);
  virtual ViewCombination matchFrames(Frame *left, Frame *right);
};

#endif
