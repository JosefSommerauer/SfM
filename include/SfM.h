/*
 * SfM.h
 *
 *  Created on: Sep 12, 2016
 *      Author: josef
 */

#ifndef SFM_H_
#define SFM_H_

#include "IFeatureDetect.h"
#include "IPoseEstimation.h"
#include "ITriangulation.h"
#include "Camera.h"
#include "ViewCombination.h"
#include "Frame.h"

#include "PointCloud.h"

typedef struct {
	double feature_detection;
	double pose_estimation;
	double triangulation;
} SfMTimes;

namespace JS {

class SfM {
public:
	SfM(IFeatureDetect  *fdetect, IPoseEstimation *poseestim, ITriangulation  *triang, Camera cam)
		: mFdetect(fdetect), mPoseestim(poseestim), mTriang(triang), mCam(cam) {
		initial_pose_estimated = false;
	}
	virtual ~SfM();

	bool InitPoseEstimation(Camera const &cam, ViewCombination &view,
			IPoseEstimation *poseestim, bool show_points, bool debug = false);


	bool find_2D_3D_correspndences(Frame *now_frame,
			std::vector<cv::Point2f> &points_2d,
			std::vector<cv::Point3f> &points_3d);

	bool Process(Frame *frame_last,	Frame *frame_now, SfMTimes &times,
			bool export_kp, bool show_matches, bool show_points, bool use_bundle_adjuster);

	bool isPoseEstimated() {return initial_pose_estimated;}

	void RunVisualization(std::string name) {mGlobalPoints.RunVisualization(name);}

	PointCloud& GetPointCloud() {return mGlobalPoints;}

	void PrintMatchMatrix();

	void GenerateTracks();

private:
	IFeatureDetect  *mFdetect;
	IPoseEstimation *mPoseestim;
	ITriangulation  *mTriang;
	Camera mCam;
	std::vector<Frame *> mFrames;
	std::vector<std::vector<std::vector<cv::DMatch> > > mMatchMatrix;

	std::vector<std::vector<int > > mTracks;

	PointCloud mGlobalPoints;
	bool initial_pose_estimated = false;
};

ITriangulation *getTriangulation(std::string name);
IFeatureDetect *getFeatureDetector(std::string name, bool cross_matching);

} /* namespace JS */
#endif /* SFM_H_ */
