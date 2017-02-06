/*
 * SfM.cpp
 *
 *  Created on: Sep 12, 2016
 *      Author: josef
 */

#include <algorithm>
#include "SfM.h"
#include "Frame.h"
#include "IterativeLinearLS.h"
#include "GPUIterativeLinearLS.h"
#include "GPUSurf.h"
#include "GPUOrb.h"
#include "IVideoStream.h"
#include "LiveStream.h"
#include "FileStream.h"
#include "OpenCVFeatureDetector.h"
#include "OpenCVPoseEstimation.h"
//#include "options.h"
#include "RubikManualFeatures.h"
#include "ViewCombination.h"
#include "OpticalFlowFeatures.h"
#include "Camera.h"
#include "StopWatch.h"
#include "FileWriter.h"
#include "tests.h"

//#include <BundleAdjuster.h>
#include <opencv2/xfeatures2d.hpp>

#include <vector>

namespace JS {

std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches) {
	std::vector<cv::DMatch> flip;
	for(int i=0;i<matches.size();i++) {
		flip.push_back(matches[i]);
		std::swap(flip.back().queryIdx,flip.back().trainIdx);
	}
	return flip;
}

SfM::~SfM() {
	for (int i = 0; i < mFrames.size(); ++i) {
		//delete mFrames[i];
	}
}

bool SfM::InitPoseEstimation(Camera const &cam, ViewCombination &view,
		IPoseEstimation *poseestim, bool show_points, bool debug) {
	cv::Mat P_0, P_1;
	PointCloud pts;
	cv::Mat K = cam.getCalibrationMatrix().clone();


	if (!poseestim->estimatePose(cam, view, P_0, P_1, pts)) {
		//std::cout << "__FILE__ line number: __LINE__" << std::endl;
		return false;
	}

	if (debug) {
		std::cout << "Initial Pose estimation successful" << std::endl;
		std::cout << "P_0" << std::endl << P_0 << std::endl;
		std::cout << "P_1" << std::endl << P_1 << std::endl;
		std::cout << "Triangulated " << pts.size() << " 3d points" << std::endl;
	}

	if (show_points) {
		pts.RunVisualization("Initial pose estimation");
	}

	std::vector<std::pair<size_t, double>> minimalErrors;
	for (size_t i = 0; i < pts.size(); i++) {
		minimalErrors.emplace_back(i, pts.getError(i));
	}
	std::sort(minimalErrors.begin(), minimalErrors.end(),
			[](std::pair<size_t, double> const &lhs, std::pair<size_t, double> const &rhs)
			{
				return lhs.second < rhs.second;
			});

	std::vector<cv::Point2f> img_points_left;
	std::vector<cv::Point2f> img_points_right;
	std::vector<cv::Point3f> points_3d;

	//std::cout << "minimalErrors.size():" << minimalErrors.size() << std::endl;

	for (size_t i = 0;
			(i < minimalErrors.size()) && (minimalErrors[i].second <= 50.0);
			i++) {
		size_t idx = minimalErrors[i].first;

		//std::cout << "minimalErrors[i].second:" << minimalErrors[i].second<< std::endl;

		if (pts.getPoint(idx)(3) <= 0)
			continue;

		img_points_left.emplace_back(view.getMatchingPointLeft(idx));
		img_points_right.emplace_back(view.getMatchingPointRight(idx));
		cv::Vec4d const &pt = pts.getPoint(idx);
		points_3d.emplace_back(pt(0), pt(1), pt(2));
	}

	if (debug) {
		std::cout << "found " << points_3d.size() << " points elegible for PnP"
				<< std::endl;
	}

	if (points_3d.size() <= 6) {
		//std::cout << "main 194 - nr. points_3d:" << points_3d.size() << std::endl;
		return false;
	}

	try {
		if (debug) {
			std::cout << "PnP of left view" << std::endl;
		}
		poseestim->estimatePose(cam, *view.getImageLeft(), points_3d,
				img_points_left);

		if (debug) {
			std::cout << "PnP of right view" << std::endl;
		}
		poseestim->estimatePose(cam, *view.getImageRight(), points_3d,
				img_points_right);
	} catch (std::exception e) {
		std::cerr << "PnP after initial pose estimation failed: " << e.what();
		return false;
	}

	return true;
}

bool SfM::find_2D_3D_correspndences(Frame *now_frame,
		std::vector<cv::Point2f> &points_2d,
		std::vector<cv::Point3f> &points_3d) {
	const double ERROR_THRESHOLD = 50;//20;//50.0; //10;
	const double DISTANCE_THRESHOLD = 50;//5.0; //0.3;

	points_2d.clear();
	points_3d.clear();

	size_t src_idx = mFrames.size() - 1;
	size_t trn_idx = 0;

	for (trn_idx = 0; trn_idx < (mFrames.size() - 1); ++trn_idx) {

		if(mMatchMatrix[src_idx][trn_idx].size() == 0) {
			ViewCombination view = mFdetect->matchFrames(mFrames[trn_idx], mFrames[src_idx]);

			std::vector<cv::DMatch> matches = view.getMatches();
			mMatchMatrix[src_idx][trn_idx] = matches;
			mMatchMatrix[trn_idx][src_idx] = FlipMatches(matches);
		}

		std::vector<cv::DMatch> matches = mMatchMatrix[src_idx][trn_idx];

		for (cv::DMatch &m : matches) {
			int idx_3d = mGlobalPoints.findCorresponding3D(mFrames[trn_idx], m.queryIdx);
			if (idx_3d != -1) {

				if (!mGlobalPoints.isViewable(idx_3d)) {
					//std::cout << "not viewable" << std::endl;
					continue;
				}
				if (mGlobalPoints.getError(idx_3d) > ERROR_THRESHOLD) {
					//std::cout << "error threshold:" << mGlobalPoints.getError(idx_3d) << std::endl;
					continue;
				}
				if (m.distance > DISTANCE_THRESHOLD) {
					//std::cout << "distance threshold:" << m.distance << std::endl;
					continue;
				}

				points_2d.push_back(now_frame->getImagepoint(m.trainIdx));
				points_3d.push_back(mGlobalPoints.getPoint3D(idx_3d));
				/*
				 std::cout << "found 2D-3D correspondence: " << points_2d.back() << " -->"
				 << points_3d.back()
				 << " error=" << pts.getError(idx_3d)
				 << std::endl;
				 */
			} else {

			}
		}
	}

	if (points_2d.size() > 5) {
		return true;
	}

	std::cout << "Not enough 2D Points:" << points_2d.size()  << " in Frame:" << mFrames.size() << std::endl;

	return false;
}

bool SfM::Process(Frame *frame_last,Frame *frame_now, SfMTimes &times,
		bool export_kp, bool show_matches, bool show_points, bool use_bundle_adjuster) {

	assert(frame_last != NULL && frame_now != NULL);

	if(mFrames.size() == 0) {
		mFrames.push_back(frame_last);
	}

	mFrames.push_back(frame_now);

	double const DISTANCE_THRESHOLD = 5;

	StopWatch fdsw;
	mFdetect->findFeatures(frame_now);
	ViewCombination view = mFdetect->matchFrames(frame_last, frame_now);

	mMatchMatrix.resize(mFrames.size());



	for (int i = 0; i < mMatchMatrix.size(); ++i) {
		mMatchMatrix[i].resize(mFrames.size());
	}

	size_t src_idx = mFrames.size() - 1;
	size_t trn_idx = mFrames.size() - 2;
	std::vector<cv::DMatch> matches = view.getMatches();
	mMatchMatrix[src_idx][trn_idx] = matches;
	mMatchMatrix[trn_idx][src_idx] = FlipMatches(matches);

	((OpenCVPoseEstimation*) mPoseestim)->setRansacParameter(cv::FM_RANSAC, 1.0,	0.99);

	if (export_kp) {
		export_keypoints(view);
	}

	if (show_matches) {
		//test_show_features_colored(view, view.getMatchesSize(), true);
	}

	double average_distance = 0.0;
	for (size_t i = 0; i < view.getMatchesSize(); i++) {
		double distance = ITriangulation::norm(view.getMatchingPointLeft(i),
				view.getMatchingPointRight(i));
		average_distance += distance;
	}
	average_distance /= view.getMatchesSize();
	fdsw.stop();
	std::cout << "average distance between " << view.getMatchesSize()
			<< " matches: " << average_distance << std::endl;

	if (average_distance < DISTANCE_THRESHOLD) {
		std::cerr << "camera has not moved" << std::endl;
		return false;
	}

	StopWatch pesw;
	if (!initial_pose_estimated) {
		std::cout << "estimating initial pose" << std::endl;
		if (!InitPoseEstimation(mCam, view, mPoseestim, show_points)) {
			std::cerr << "pose estimation failed" << std::endl;
			return false;
		} else {
			initial_pose_estimated = true;
		}
	} else {
		std::vector<cv::Point2f> points_2d;
		std::vector<cv::Point3f> points_3d;

		std::cout << "finding 2D 3D correspondences" << std::endl;
		if (!find_2D_3D_correspndences(frame_now, points_2d, points_3d)) {
			std::cerr << "could not find enough correspondences" << std::endl;
			return false;
		}

		std::cout << "starting PnP" << std::endl;
		if (!mPoseestim->estimatePose(mCam, *frame_now, points_3d, points_2d)) {
			std::cerr << "PnP failed" << std::endl;
			return false;
		}
	}

	//((OpenCVPoseEstimation*)mPoseestim)->removeOutlier(mCam,view);

	pesw.stop();

	std::cout << "starting triangulation of " << view.getMatchesSize()
			<< " keypoints" << std::endl;
	StopWatch trsw;
	if (!mTriang->triangulate(mCam, view, mGlobalPoints)) {
		std::cerr << "triangulation failed" << std::endl;
		return false;
	}

	// remove 3D point with high error rate
	mGlobalPoints.getPoints3D();

	trsw.stop();
	/*
	 OpticalFlowFeatures of;
	 std::cout << "starting of" << std::endl;
	 ViewCombination vof = of.matchFrames(frame_last, frame_now);
	 std::cout << "matches:  " << vof.getMatchesSize() << std::endl
	 << "kp left:  " << frame_last->getKeypointsSize() << std::endl
	 << "kp right: " << frame_now->getKeypointsSize() << std::endl;
	 std::cout << "starting triangulating of" << std::endl;
	 triang->triangulate(cam, vof,
	 view.getImageLeft()->getProjectionMatrix(),
	 view.getImageRight()->getProjectionMatrix(),
	 global_points);
	 //of_pts.RunVisualization();
	 */
	/*
	 OpticalFlowFeatures of;
	 std::cout << "starting of" << std::endl;
	 Frame f1(img_last);
	 Frame f2(img_now);
	 ViewCombination vof = of.matchFrames(&f1, &f2);
	 triang->triangulate(cam, vof,
	 frame_last->getProjectionMatrix(),
	 frame_now->getProjectionMatrix(),
	 global_points);
	 */

	// bundle adjustment

	if(use_bundle_adjuster) {
		std::cout << "bundle adjustment is not implemented!" << std::endl;
	/*
		BundleAdjuster mBundleAjuster;

		std::vector<CloudPoint> new_pountcloud, old_pountcloud;
		std::vector<std::vector<cv::KeyPoint> > new_imgpts;
		std::map<int, cv::Matx34d> new_Pmats;
		int framecnt = 0;
		cv::Mat K = mCam.getCalibrationMatrix();

		for (int i = 0; i < mFrames.size(); ++i) {
			new_imgpts.push_back(mFrames[i]->getKeypoints());
			new_Pmats.insert(std::make_pair(i,mFrames[i]->getProjectionMatrix()));
		}

		for (int i = 0; i < mGlobalPoints.size(); ++i) {
			CloudPoint tmp_cp;
			tmp_cp.pt.x = mGlobalPoints.getPoint(i).val[0];
			tmp_cp.pt.y = mGlobalPoints.getPoint(i).val[1];
			tmp_cp.pt.z = mGlobalPoints.getPoint(i).val[2];
			tmp_cp.imgpt_for_img = mGlobalPoints.getImgptForImg(i);

			new_pountcloud.push_back(tmp_cp);
		}

		// find unkown correspondences

//		for (int i = 0; i < new_pountcloud.size(); ++i) {
//			int j;
//			for (j = 0; j <  new_pountcloud[i].imgpt_for_img.size(); ++j) {
//				if(new_pountcloud[i].imgpt_for_img[j] == -1) {
//					break;
//				}
//			}
//
//			for(int k=0; k  < new_pountcloud.size(); ++k ) {
//				for (int l = 0; l < new_pountcloud[k].imgpt_for_img.size(); ++l) {
//					if(k != 1 && (new_pountcloud[i].imgpt_for_img[l] == new_pountcloud[k].imgpt_for_img[l])
//							&& new_pountcloud[k].imgpt_for_img[l] != -1) {
//						for (int m = 0; m < new_pountcloud[k].imgpt_for_img.size(); ++m) {
//							if((new_pountcloud[k].imgpt_for_img[m] != -1)
//							&& (new_pountcloud[i].imgpt_for_img[m] == -1)) {
//								new_pountcloud[i].imgpt_for_img[m] = new_pountcloud[k].imgpt_for_img[m];
//							}
//						}
//					}
//				}
//			}
//
//		}
		//

		// sort point cloud


		for (int i = 0; i < new_pountcloud.size(); ++i) {
			std::cout << new_pountcloud[i].pt << ";";
			for (int j = 0; j <  new_pountcloud[i].imgpt_for_img.size(); ++j) {
				std::cout << new_pountcloud[i].imgpt_for_img[j] << ";";
			}
			std::cout << std::endl;
		}

		// ---------------------------------------------------------------------------------

		//exit(0);

		old_pountcloud.resize(new_pountcloud.size());
		std::copy(new_pountcloud.begin(),new_pountcloud.end(), old_pountcloud.begin());

		std::cout << "3d points befor ba:" << new_pountcloud.size() << std::endl;

		mBundleAjuster.adjustBundle(new_pountcloud,K,new_imgpts,new_Pmats);

		std::cout << "3d points after ba:" << new_pountcloud.size() << std::endl;

		//mCam.setCalibrationMatrix(K);

		//mGlobalPoints.clearPoints();
		std::vector<cv::Vec4d> newPtCl;
		for (int i = 0; i < new_pountcloud.size(); ++i) {
			cv::Vec4d tmp_vec;
			tmp_vec[0] = new_pountcloud[i].pt.x;
			tmp_vec[1] = new_pountcloud[i].pt.y;
			tmp_vec[2] = new_pountcloud[i].pt.z;
			tmp_vec[3] = 1.0;
			//mGlobalPoints.getPoint(i) = tmp_vec;
			//mGlobalPoints.addPoint()
			newPtCl.push_back(tmp_vec);
		}
		if(mGlobalPoints.UpdatePoints(newPtCl) == false) {
			std::cout << "error update pointcloud" << std::endl;
			exit(0);
		}


		for (auto i = new_Pmats.begin(); i != new_Pmats.end(); ++i) {
			cv::Mat p_mat(i->second);

			mFrames[i->first]->setProjectionMatrix(p_mat);
		}
		*/
	}

	// -- end bandle ajustment --------------------------------------- //

	if (show_points) {
		mGlobalPoints.RunVisualization("Global points in sfm loop");
	}

	times.feature_detection = fdsw.getDurationMs();
	times.pose_estimation = pesw.getDurationMs();
	times.triangulation = trsw.getDurationMs();
	return true;
}

IFeatureDetect *getFeatureDetector(std::string name, bool cross_matching) {
	double hessian = 800;
	double octaves = 5;
	int layers = 10;
	bool extended = true;
	bool upright = true; //false;

	/*
	const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
	const double ransac_thresh = 2.5f; // RANSAC inlier threshold
	const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
	const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box
	const int stats_update_period = 10; // On-screen statistics are updated every 10 frames
	*/

	std::transform(name.begin(), name.end(),name.begin(), ::toupper);

	if (name == "RUB") {
		return new RubikManualFeatures();
	} else if (name == "OF") {
		return new OpticalFlowFeatures();
	} else if (name == "GPU") {
		return new GPUSurf(hessian, octaves, layers, extended, upright, cv::NORM_L2);
	} else if (name == "SIFT") {
		OpenCVFeatureDetector *ocvfeat = new OpenCVFeatureDetector(cross_matching);
		ocvfeat->createExtractor<cv::xfeatures2d::SIFT>();
		ocvfeat->createDetector<cv::xfeatures2d::SIFT>();
		//ocvfeat->createMatcher<cv::BFMatcher>(cv::NORM_L2, true);
		ocvfeat->createMatcher<cv::FlannBasedMatcher>();
		return ocvfeat;
	} else if (name == "SURF") {
		OpenCVFeatureDetector *ocvfeat = new OpenCVFeatureDetector(cross_matching);
		ocvfeat->createExtractor<cv::xfeatures2d::SURF>(hessian, octaves, layers, extended, upright);
		ocvfeat->createDetector<cv::xfeatures2d::SURF>(hessian, octaves, layers, extended, upright);
		//ocvfeat->createDetector<cv::GFTTDetector>(1000,0.01,10,3,false,0.04);
		//ocvfeat->createMatcher<cv::BFMatcher>(cv::NORM_L2, true);
		ocvfeat->createMatcher<cv::FlannBasedMatcher>();
		return ocvfeat;
	} else if (name == "BRIEF") {
		OpenCVFeatureDetector *ocvfeat = new OpenCVFeatureDetector(cross_matching);
		ocvfeat->createExtractor<cv::xfeatures2d::BriefDescriptorExtractor>();
		ocvfeat->createDetector<cv::FastFeatureDetector>();
		ocvfeat->createMatcher<cv::BFMatcher>(cv::NORM_HAMMING, cross_matching);
		return ocvfeat;
	} else if (name == "ORB") {
		OpenCVFeatureDetector *ocvfeat = new OpenCVFeatureDetector(cross_matching);
		ocvfeat->createExtractor<cv::ORB>();
		ocvfeat->createDetector<cv::ORB>();
		ocvfeat->createMatcher<cv::BFMatcher>(cv::NORM_HAMMING, cross_matching);
		return ocvfeat;
	} else if (name == "ORB5000") {
		OpenCVFeatureDetector *ocvfeat = new OpenCVFeatureDetector(cross_matching);
		ocvfeat->createExtractor<cv::ORB>(5000);
		ocvfeat->createDetector<cv::ORB>(5000);
		ocvfeat->createMatcher<cv::BFMatcher>(cv::NORM_HAMMING, cross_matching);
		return ocvfeat;
	} else if (name == "ORBGPU") {
		return new GPUOrb();
	} else if (name == "ORB5000GPU") {
		return new GPUOrb(5000);
	} else if (name == "AKAZE") {
		std::cout << "hallo world"<< std::endl;
	
		OpenCVFeatureDetector *ocvfeat = new OpenCVFeatureDetector(cross_matching);
		ocvfeat->createExtractor<cv::AKAZE>();
		ocvfeat->createDetector<cv::AKAZE>();
				
		ocvfeat->createMatcher<cv::BFMatcher>(cv::NORM_HAMMING, cross_matching);
		return ocvfeat;
	} else if (name == "KAZE") {
		OpenCVFeatureDetector *ocvfeat = new OpenCVFeatureDetector(cross_matching);
		ocvfeat->createExtractor<cv::KAZE>();
		ocvfeat->createDetector<cv::KAZE>();
		//ocvfeat->createMatcher<cv::BFMatcher>(cv::NORM_L2, cross_matching);
		ocvfeat->createMatcher<cv::FlannBasedMatcher>();
		return ocvfeat;

		//


	}

	return nullptr;
}


ITriangulation *getTriangulation(std::string name) {
	if (name == "GPU") {
		return new GPUIterativeLinearLS(100);
	} else if (name == "OCV") {
		return new IterativeLinearLS<double>();
	}

	return nullptr;
}

void SfM::PrintMatchMatrix() {

	std:: cout << std::string(80,'-') << std::endl;
	std::cout << "MatchMatrix:" << std::endl;
	for (int i = 0; i < mMatchMatrix.size(); ++i) {
		for (int j = 0; j < mMatchMatrix[i].size(); ++j) {
			std::cout << mMatchMatrix[i][j].size() << '\t';
		}
		std:: cout << std::endl;
	}
	std:: cout << std::string(80,'-') << std::endl;

	/*
	for (int i = 0; i < mMatchMatrix.size() - 1; ++i) {
		for (int j = 0; j < mMatchMatrix[i].size(); ++j) {
			if(i == j) continue; // dont match with it self

			std::cout << "Matches:" << i << "-" << j << std::endl;

			for (int k = 0; k < mMatchMatrix[i][j].size(); ++k) {
				// match 0-1

				int newQueryIdx;
				newQueryIdx = mMatchMatrix[i][j][k].queryIdx;
				std::cout << mMatchMatrix[i][j][k].queryIdx << " -> ";
				std::cout << mMatchMatrix[i][j][k].trainIdx << ";";

				// match 1-2
				for (int m = j+1; m < mMatchMatrix[i].size(); ++m) {
					bool found = false;
					for (int n = 0; n < mMatchMatrix[i][m].size(); ++n) {
						if(mMatchMatrix[i][m][n].queryIdx == newQueryIdx) {
							found = true;
							//std::cout << mMatchMatrix[i][m][n].queryIdx << " -> ";
							//std::cout << mMatchMatrix[i][m][n].trainIdx << ";";
							break;
						}
					}
					//if(!found) std::cout << "-1;";
				}
				//std::cout << std::endl;
			}
		}
		std:: cout << std::endl;
	}
	*/
}

void SfM::GenerateTracks() {
	/*
	for (int i = 0; i < mFrames.size(); ++i) {
		mTracks.push_back(std::vector<int>());

		for (int j = 0; j < mFrames.size(); ++j) {
			std::vector<cv::DMatch> matches = mMatchMatrix[i][j];
			for (int j = 0; j < matches.size(); ++j) {

				matches[j].queryIdx;

				mTracks[i].
			}
		}


		mTracks.push_back(mMatchMatrix[i][j])
	}

	*/
}

} /* namespace JS */


