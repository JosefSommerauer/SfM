#include "OpenCVPoseEstimation.h"

#include <iostream>
#include <vector>

OpenCVPoseEstimation::OpenCVPoseEstimation(ITriangulation *triangulation)
                                          : IPoseEstimation(triangulation)
{
}

OpenCVPoseEstimation::~OpenCVPoseEstimation()
{
}

void OpenCVPoseEstimation::setRansacParameter(int method, double param_1, double param_2)
{

  //std::cout << "---- OpenCV FundMat Parmas has been set " << std::endl;
  mFindFMethod = method;
  mFindFOutlierDistance = param_1;
  mFindFConfidence = param_2;
}

inline bool OpenCVPoseEstimation::isRotationMatrix(cv::Mat const &R)
{
  double det = cv::determinant(R);
  if (std::fabs(det - 1) > DET_ROT_MAT_THRESHOLD) {
    return false;
  }
  return true;
}

inline bool OpenCVPoseEstimation::checkE(cv::Mat const &E, cv::Mat const &svd_sigma)
{
  if (std::fabs(cv::determinant(E)) > DET_E_THRESHOLD) {
    std::cerr << "determinant of E is not feasable: " << std::endl
              << cv::determinant(E) << std::endl;
    return false;
  }

  // singular values of E have to be about equal
  double const s_1 = svd_sigma.at<double>(0, 0);
  double const s_2 = svd_sigma.at<double>(0, 0);
  double sing_ratio = s_1 / s_2;
  if (std::fabs(sing_ratio - 1) > SING_RATIO_E_THRESHOLD) {
    std::cerr << "Singular value of E is not feasable: " << std::endl
              << s_1 << " and " << s_2 << std::endl;
    return false;
  }

  return true;
}

void OpenCVPoseEstimation::removeOutlier(Camera const & cam, ViewCombination &view) {
	  cv::Mat OutlierMask, R, t;
	  cv::Mat E = cv::findEssentialMat(view.getMatchingPointsLeft(), view.getMatchingPointsRight(), cam.getCalibrationMatrix(),
			  	  	  	  	  	  	  	  mFindFMethod, mFindFConfidence, mFindFOutlierDistance, OutlierMask);

	  std::vector<cv::DMatch> oldMatches, newMatches;
	  oldMatches = view.getMatches();

	  for (int i = 0; i < OutlierMask.rows; ++i) {
		  if(OutlierMask.at<int>(i)!=0) {
			  newMatches.push_back(oldMatches[i]);
		  }
	  }
	  view.setMatches(newMatches);
}

bool OpenCVPoseEstimation::estimatePose(Camera const & cam, ViewCombination &view,
                                        cv::Mat &P_0, cv::Mat &P_1, PointCloud &pts)
{
  assert(mTriangulation != NULL);
  if (view.getMatchesSize() < 8) {
    std::cerr << "not enough keypoints" << std::endl;
    return false;
  }

  P_0 = cv::Mat(cv::Matx34d::eye());

  	  /*
	std::cout << " mFindFMethod: " << mFindFMethod 
				 << " mFindFOutlierDistance: " << mFindFOutlierDistance 
				 << " mFindFConfidence: " << mFindFConfidence << std::endl;
	*/

  /*
  cv::Mat F = cv::findFundamentalMat(view.getMatchingPointsLeft(), view.getMatchingPointsRight(),
                                     mFindFMethod, mFindFOutlierDistance, mFindFConfidence);
  cv::Mat E = cam.getCalibrationMatrix().t() * F * cam.getCalibrationMatrix();

  */

  cv::Mat OutlierMask, R, t;
  cv::Mat E = cv::findEssentialMat(view.getMatchingPointsLeft(), view.getMatchingPointsRight(), cam.getCalibrationMatrix(),
		  	  	  	  	  	  	  	  mFindFMethod, mFindFConfidence, mFindFOutlierDistance, OutlierMask);

#if 1 // OpenCV Pose Recovery
  cv::recoverPose(E, view.getMatchingPointsLeft(), view.getMatchingPointsRight(),
		  cam.getCalibrationMatrix(), R, t, OutlierMask);
  cv::hconcat(R, t, P_1);

  std::vector<cv::DMatch> old_Matches = view.getMatches();
  std::vector<cv::DMatch> new_Matches;


  assert(old_Matches.size() == OutlierMask.rows);

  // remove outlier
  for (int i = 0; i < OutlierMask.rows; ++i) {
	  if(OutlierMask.at<bool>(i)) {
		  new_Matches.push_back(old_Matches[i]);
	  }
  }

  view.setMatches(new_Matches);

  pts.clearPoints();
  view.getImageLeft()->setProjectionMatrix(P_0);
  view.getImageRight()->setProjectionMatrix(P_1);
  mTriangulation->triangulate(cam, view, pts);
#else // zoble pose recover Pose


  /*
  std::vector<cv::DMatch> oldMatches, newMatches;
  oldMatches = view.getMatches();

  std::cout << "Matches with Outliers: " << oldMatches.size() << ";" << OutlierMask.rows << std::endl;

  for (int i = 0; i < OutlierMask.rows; ++i) {
	  if(OutlierMask.at<int>(i) != 0) {
		  newMatches.push_back(oldMatches[i]);
	  }
  }

  std::cout << "Matches without Outliers: " << newMatches.size() << std::endl;
  view.setMatches(newMatches);
  */


  cv::SVD svd(E);

  if (!checkE(E, svd.w)) {
    return false;
  }

  //either U or Vt has to be a rotation matrix otherwise flip sign of U
  if (isRotationMatrix(svd.u) != isRotationMatrix(svd.vt)) {
    svd.u = -svd.u;
  }

  // construct rotation and translation solutions
  cv::Mat_<double> const R90 = (cv::Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
  
  cv::Mat_<double> R1 = svd.u * R90 * svd.vt;
  cv::Mat_<double> R2 = svd.u * R90.t() * svd.vt;
  cv::Mat_<double> t1 = svd.u.col(2);
  cv::Mat_<double> t2 = -svd.u.col(2);

  if (!isRotationMatrix(R1) || !isRotationMatrix(R2)) {
    std::cerr << "R1 or R2 is not a rotation matrix" << std::endl;
    return false;
  }


  size_t comb = 1;
  for (comb = 1; comb <= 4; comb++) {
    switch (comb) {
      case 1:
        P_1 = (cv::Mat_<double>(3, 4) << R1(0,0), R1(0,1), R1(0,2), t1(0),
                                         R1(1,0), R1(1,1), R1(1,2), t1(1),
                                         R1(2,0), R1(2,1), R1(2,2), t1(2));
        break;
      case 2:
        P_1 = (cv::Mat_<double>(3, 4) << R1(0,0), R1(0,1), R1(0,2), t2(0),
                                         R1(1,0), R1(1,1), R1(1,2), t2(1),
                                         R1(2,0), R1(2,1), R1(2,2), t2(2));
        break;
      case 3:
        P_1 = (cv::Mat_<double>(3, 4) << R2(0,0), R2(0,1), R2(0,2), t1(0),
                                         R2(1,0), R2(1,1), R2(1,2), t1(1),
                                         R2(2,0), R2(2,1), R2(2,2), t1(2));
        break;
      case 4:
        P_1 = (cv::Mat_<double>(3, 4) << R2(0,0), R2(0,1), R2(0,2), t2(0),
                                         R2(1,0), R2(1,1), R2(1,2), t2(1),
                                         R2(2,0), R2(2,1), R2(2,2), t2(2));
        break;
      default:
        assert(false);
    }


#ifdef POSE_PRINT
    std::cout << "triangulating points for comb " << comb << std::endl;
#endif
    pts.clearPoints();
    view.getImageLeft()->setProjectionMatrix(P_0);
    view.getImageRight()->setProjectionMatrix(P_1);
    mTriangulation->triangulate(cam, view, pts);


    /*
    for (size_t i = 0; i < pts.size(); i++) {
      std::cout << "error[" << i << "] = " << pts.getError(i) << std::endl;
    }
    //TODO test P_1, P_0 as well?

    std::cout << "average error " << pts.getAverageError() << std::endl;

    if (pts.getAverageError() > AVG_REPROJECTION_ERROR_THRESHOLD) {
      continue;
    }
    */

    if (!testChirality(pts, P_0) ||
        !testChirality(pts, P_1)) {
      continue;
    }

#ifdef POSE_PRINT
    std::cout << "Found P1 in combination " << comb << std::endl;
#endif
    break;
  }

  if (comb > 4) {
    std::cerr << "No combination yielded a useable P1" << std::endl;
    return false;
  }
#endif
  return true;
}

bool OpenCVPoseEstimation::estimatePose(Camera const &cam, Frame &frame,
                                        std::vector<cv::Point3f> const &known_3d,
                                        std::vector<cv::Point2f> const &known_2d)
{
  assert(known_3d.size() != 0);
  assert(known_3d.size() == known_2d.size());

  cv::Mat dist_coeff = cam.getDistortionCoefficients();
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat inliers;

  bool success = cv::solvePnPRansac(known_3d, known_2d, cam.getCalibrationMatrix(), dist_coeff,
                                    rvec, tvec,
                                    false, 100, mFindFOutlierDistance, mFindFConfidence, inliers, cv::SOLVEPNP_EPNP);

  /*
  for (int i = 0; i < inliers.rows; ++i) {
	std::cout << inliers.at<int>(i) << std::endl;
  }
  std::cout << "number of inliers:" << inliers.rows << std::endl;
  */

  cv::Mat R;
  cv::Rodrigues(rvec, R);
  if (!isRotationMatrix(R)) {
    std::cerr << "R is not a rotation matrix" << std::endl;
    return false;
  }

  cv::Mat P;
  cv::hconcat(R, tvec, P);

  frame.setProjectionMatrix(P);

  return success;
}

