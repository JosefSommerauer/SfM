/*
 * StaticAllocator.h
 *
 *  Created on: Dec 14, 2016
 *      Author: josef
 */

#ifndef STATICALLOCATOR_H_
#define STATICALLOCATOR_H_

#include "IFeatureDetect.h"
#include "GPUHelper.h"

#include <opencv2/xfeatures2d/cuda.hpp>
#include <cuda_runtime.h>

class StaticAllocator : public cv::cuda::GpuMat::Allocator {

  private:

    unsigned char *mData = nullptr;
    size_t mLen = 0;
    size_t mStep = 0;

    void allocate(int rows, int cols, size_t elemSize)
    {
      mLen = elemSize * cols * rows;
      if (rows > 1 && cols > 1)
      {
        GPUHelper::check(cudaMallocPitch(&mData, &mStep, elemSize * cols, rows));
      }
      else
      {
        //Single row or single column must be continuous
        GPUHelper::check(cudaMalloc(&mData, mLen));
        mStep = elemSize * cols;
      }
    }

    void free()
    {
      if (mData != nullptr) {
        GPUHelper::check(cudaFree(mData));
      }
      mData = nullptr;
      mLen = 0;
    }

  public:

  ~StaticAllocator()
  {
    this->free();
  }

  virtual bool allocate(cv::cuda::GpuMat *mat, int rows, int cols, size_t elemSize)
  {
    if ((mData == nullptr) || ((rows * cols * elemSize) > mLen)) {
      this->free();
      this->allocate(rows, cols, elemSize);
    }

    mat->data = mData;
    mat->step = mStep;
    mat->refcount = new int;
    return true;
  }

  virtual void free(cv::cuda::GpuMat *mat)
  {
    delete mat->refcount;
  }

};



#endif /* STATICALLOCATOR_H_ */
