#include "GPUOrb.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d/cuda.hpp>

GPUOrb::GPUOrb(int nfeatures)
	:mNorm(cv::NORM_HAMMING)
{
	f2d = cv::cuda::ORB::create(nfeatures);
}

bool GPUOrb::findFeatures(Frame *frame)
{
  cv::cuda::GpuMat gpu_image(&mImageAllocator);
  cv::cuda::GpuMat gpu_keypoints(&mKeypointsAllocator);
  cv::cuda::GpuMat gpu_mask(&mMaskAllocator);
  cv::cuda::Stream mStream;
  gpu_image.upload(frame->getImage());
  //mSurf(gpu_image, gpu_mask, gpu_keypoints, frame->getGpuDescriptors());
  f2d->detectAndComputeAsync(gpu_image, gpu_mask,
			gpu_keypoints, frame->getGpuDescriptors(), false, mStream); // true = useProvidedKeypoints
  mStream.waitForCompletion();

  std::vector<cv::KeyPoint> keypoints;
  //mSurf.downloadKeypoints(gpu_keypoints, keypoints);
  f2d->convert(gpu_keypoints, keypoints);
  frame->setKeypoints(keypoints);

  return true;
}

ViewCombination GPUOrb::matchFrames(Frame *left, Frame *right)
{
	//cv::cuda::BFMatcher_CUDA matcher(mNorm);
	//cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(mNorm);

	cv::Ptr<cv::cuda::DescriptorMatcher> matcher_gpu = cv::cuda::DescriptorMatcher::createBFMatcher(mNorm);

	std::vector<cv::DMatch> matches, good_matches;
	std::vector<std::vector<cv::DMatch> > matches12, matches21;

	ViewCombination view(left, right);

	/*matcher->match(left->getGpuDescriptors(), right->getGpuDescriptors(), matches);*/

	int knn = 2;
	//matcher->knnMatch( left->getGpuDescriptors(), right->getGpuDescriptors(), matches12, knn );

	matcher_gpu->knnMatch( left->getGpuDescriptors(), right->getGpuDescriptors(), matches12, knn );

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
