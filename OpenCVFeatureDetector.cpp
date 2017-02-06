#include "OpenCVFeatureDetector.h"

#include <iostream>
#include <set>

#define PI 3.14159265

OpenCVFeatureDetector::~OpenCVFeatureDetector()
{
}

bool OpenCVFeatureDetector::findFeatures(Frame *frame)
{
  assert(!mDetector.empty());
  assert(!mExtractor.empty());

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  mDetector->detect(frame->getImage(), keypoints);
  mExtractor->compute(frame->getImage(), keypoints, descriptors);

  frame->setKeypoints(keypoints);
  frame->setDescriptors(descriptors);

  return true;
}

/* simple matching
ViewCombination OpenCVFeatureDetector::matchFrames(Frame *left, Frame *right)
{
  assert(!mMatcher.empty());
  ViewCombination view(left, right);

  std::vector<cv::DMatch> matches, good_matches;

  mMatcher->clear();
  mMatcher->add(left->getDescriptors());

  mMatcher->match(left->getDescriptors(), right->getDescriptors(), matches);

  filterMatches(matches, left->getKeypointsSize(), right->getKeypointsSize(), good_matches);

  view.setMatches(good_matches);

  return view;
}
*/

void getMinMaxDistance(std::vector<cv::DMatch> matches,
		double & min_dist, double & max_dist)
{
	max_dist = 0;
	min_dist = 100;

	for( int i = 0; i < matches.size(); i++ )
  {
		double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
}

std::vector<cv::DMatch> distanceTest(std::vector<cv::DMatch> matches) {
	std::vector<cv::DMatch > good_matches;

	double max_dist = 0; double min_dist = 10000;
	getMinMaxDistance(matches, min_dist, max_dist);

	for( int i = 0; i < matches.size(); i++ ) {
		if(matches[i].distance <= cv::max(10*min_dist, 0.02)){
			good_matches.push_back(matches[i]);
		}
	}

	return good_matches;
}

// cross matching
ViewCombination OpenCVFeatureDetector::matchFrames(Frame *left, Frame *right)
{
	assert(!mMatcher.empty());
	ViewCombination view(left, right);
	
	int knn;

	std::vector<std::vector<cv::DMatch> > matches12, matches21;
	std::vector<cv::DMatch> better_matches, good_matches, matches;

	mMatcher->clear();

	if(mUseCrossMatching) {
		knn = 1;
		mMatcher->knnMatch( left->getDescriptors(), right->getDescriptors(), matches12, knn );
		mMatcher->knnMatch( right->getDescriptors(), left->getDescriptors(), matches21, knn );

	  //mMatcher->add(left->getDescriptors());

	  //mMatcher->match(left->getDescriptors(), right->getDescriptors(), matches);

		for( size_t m = 0; m < matches12.size(); m++ ) {
			bool findCrossCheck = false;
			for( size_t fk = 0; fk < matches12[m].size(); fk++ ) {
				cv::DMatch forward = matches12[m][fk];

				for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ ) {
					 cv::DMatch backward = matches21[forward.trainIdx][bk];
					 if( backward.trainIdx == forward.queryIdx ) {
						  matches.push_back(forward);
						  findCrossCheck = true;
						  break;
					 }
				}
				if( findCrossCheck ) break;
			}
		}

	} else { // use lowe test
		knn = 2;
		mMatcher->knnMatch( left->getDescriptors(), right->getDescriptors(), matches12, knn );
		for (int i = 0; i < matches12.size(); ++i)
		{
			  const float ratio = 0.8; // As in Lowe's paper; can be tuned
			  if (matches12[i][0].distance < ratio * matches12[i][1].distance)
			  {
			      matches.push_back(matches12[i][0]);
			  }
		}
	}

	filterMatches(matches, left->getKeypointsSize(), right->getKeypointsSize(), good_matches);

	//good_matches = distanceTest(good_matches);

	/*
	double average_alpha = 0.0;

	for(int i=0; i < good_matches.size(); i++) {
		std::cout << good_matches[i].imgIdx
			<< " " << good_matches[i].distance
			<< " " << good_matches[i].queryIdx
	 		<< " " << good_matches[i].trainIdx;

		// calculate angle
		cv::Point2f posleft  = left->getKeypoint(good_matches[i].queryIdx).pt;
		cv::Point2f posright = right->getKeypoint(good_matches[i].trainIdx).pt;

		int x = posleft.x - posright.x;
		int y = posleft.y - posright.y;
		double alpha = atan ((double)y/x) * 180 / PI;

		std::cout << " " << posleft << " " << posright << " " << alpha << std::endl; 

		if(i==0) {
			average_alpha = alpha;
		} else {
			average_alpha += alpha;
			average_alpha /= 2.0; 
		}
	}

	for(int i=0; i < good_matches.size(); i++) { 
		// calculate angle
		cv::Point2f posleft  = left->getKeypoint(good_matches[i].queryIdx).pt;
		cv::Point2f posright = right->getKeypoint(good_matches[i].trainIdx).pt;

		int x = posleft.x - posright.x;
		int y = posleft.y - posright.y;
		double alpha = atan ((double)y/x) * 180 / PI;

		if(alpha < average_alpha+20 && alpha > average_alpha-20) {
			better_matches.push_back(good_matches[i]);
		}		
	}

	std::cout << "average alpha:" << average_alpha << std::endl;
	*/
	
	view.setMatches(good_matches);
	//view.setMatches(better_matches);

	return view;
}




