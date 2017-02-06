#include "GPUSurf.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

GPUSurf::GPUSurf(double hessian, int octaves, int layers, bool extended, bool upright,
                 int norm, float ratio)
                : mSurf(hessian, octaves, layers, extended, ratio, upright), mNorm(norm)
{
}

bool GPUSurf::findFeatures(Frame *frame)
{
  cv::cuda::GpuMat gpu_image(&mImageAllocator);
  cv::cuda::GpuMat gpu_keypoints(&mKeypointsAllocator);
  cv::cuda::GpuMat gpu_mask(&mMaskAllocator);

  gpu_image.upload(frame->getImage());
  mSurf(gpu_image, gpu_mask, gpu_keypoints, frame->getGpuDescriptors());

  std::vector<cv::KeyPoint> keypoints;
  mSurf.downloadKeypoints(gpu_keypoints, keypoints);
  frame->setKeypoints(keypoints);

  return true;
}

ViewCombination GPUSurf::matchFrames(Frame *left, Frame *right)
{
	//cv::cuda::BFMatcher_CUDA matcher(mNorm);
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(mNorm);
	std::vector<cv::DMatch> matches, good_matches;
	std::vector<std::vector<cv::DMatch> > matches12, matches21;

	ViewCombination view(left, right);

	/*matcher->match(left->getGpuDescriptors(), right->getGpuDescriptors(), matches);*/

	int knn = 2;
	matcher->knnMatch( left->getGpuDescriptors(), right->getGpuDescriptors(), matches12, knn );
	for (int i = 0; i < matches12.size(); ++i)
	{
	  const float ratio = 0.8; // As in Lowe's paper; can be tuned
	  if (matches12[i][0].distance < ratio * matches12[i][1].distance)
	  {
		  matches.push_back(matches12[i][0]);
	  }
	}

	filterMatches(matches, left->getKeypointsSize(), right->getKeypointsSize(), good_matches);

	view.setMatches(good_matches);
	return view;
}
