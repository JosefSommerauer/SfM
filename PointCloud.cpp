#include "PointCloud.h"

#include <opencv2/viz.hpp>

#include <algorithm>
#include <chrono>
#include <thread>

#include <iostream>
#include <algorithm>    // std::min_element, std::max_element

#include <Eigen/Dense>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ros/conversions.h>

#include <opencv2/highgui.hpp>
#include <opencv2/viz/types.hpp>



void PointCloud::addPoint(cv::Vec4d point, double error,
                          std::pair<Frame const *, size_t> left_imgpt,
                          std::pair<Frame const *, size_t> right_imgpt)
{
	//if(error < 100.0) {
	// check if point is already in cloud

		int PointIdx = -1;


		for (int i = 0; i < mCorresponding2D[left_imgpt.first].size(); ++i) {
			if(mCorresponding2D[left_imgpt.first][i].first == left_imgpt.second) {
				PointIdx = mCorresponding2D[left_imgpt.first][i].second;
				//std::cout << "old point:" << mPoints[PointIdx] << "\terror:" << mReprojectionErrors[PointIdx]<< std::endl;
				//std::cout << "new point:" << point << "\terror:" << error << std::endl;
				//exit(0);

				// if new point is better use new point
				if(mReprojectionErrors[PointIdx] > error) {
					mPoints[PointIdx] = point;
					mReprojectionErrors[PointIdx] = error;
				}

				break;
			}
		}



		for (int i = 0; i < mCorresponding2D[right_imgpt.first].size(); ++i) {
			if(mCorresponding2D[right_imgpt.first][i].first == right_imgpt.second) {
				std::cout << "error! PointCloud.cpp - line 44: " << right_imgpt.first << std::endl;
				exit(0);
			}
		}


		// if new point
		if(PointIdx == -1) {
			mPoints.push_back(point);
			PointIdx = mPoints.size()-1;
			mReprojectionErrors.push_back(error);
		}

		std::pair<size_t, size_t> left_2d_3d = std::make_pair(left_imgpt.second, PointIdx);
		std::pair<size_t, size_t> right_2d_3d = std::make_pair(right_imgpt.second, PointIdx);
		mCorresponding2D[left_imgpt.first].push_back(left_2d_3d);
		mCorresponding2D[right_imgpt.first].push_back(right_2d_3d);
	//}
}

void PointCloud::removePoint(size_t const pt3D_idx) {
	/*
	mPoints.erase(pt3D_idx);
	mReprojectionErrors.erase(pt3D_idx);

	// delete 2D - 3D Correspondence
	int delcnt = 0;

	for (auto i = mCorresponding2D.begin(); i != mCorresponding2D.end(); ++i) {
	    for (int j = 0; j < i->second.size(); ++j) {
			if(i->second[j].second == pt3D_idx) {
				i->second.erase(j);
				delcnt++;
			}
		}
	}

	for (auto i = mCorresponding2D.begin(); i != mCorresponding2D.end(); ++i) {
	    for (int j = 0; j < i->second.size(); ++j) {
			if(i->second[j].second > pt3D_idx) {
				i->second[j].second-=delcnt;
			}
		}
	}
	*/
}

cv::Vec4d const &PointCloud::getPoint(int i) const
{
  return mPoints.at(i);
}

double const &PointCloud::getError(int i) const
{
  return mReprojectionErrors.at(i);
}

int PointCloud::findCorresponding3D(Frame const *frame, size_t const imgpt_idx) const
{
  auto iter = mCorresponding2D.find(frame);
  if (iter == mCorresponding2D.end()) {
    return -1;
  }

  std::vector<std::pair<size_t, size_t>> const &corr_2d_3d = iter->second;
  for (size_t i = 0; i < corr_2d_3d.size(); i++) {
    if (corr_2d_3d[i].first == imgpt_idx) {
      return corr_2d_3d[i].second;
    }
  }
  return -1;
}

cv::Vec3b PointCloud::getColorOf3Dform2DPoint(size_t const pt3D_idx) const
{
	for (auto i = mCorresponding2D.begin(); i != mCorresponding2D.end(); ++i) {
	    for (auto j = i->second.begin(); j < i->second.end(); ++j) {
			if(j->second == pt3D_idx) {
				cv::KeyPoint kpt = i->first->getKeypoint(j->first);
				int x = kpt.pt.x;
				int y = kpt.pt.y;
				cv::Mat img = i->first->getColorImage();
				return img.at<cv::Vec3b>(y, x);
			}
		}
	}

	std::cout << "point not found" << std::endl;
	return cv::Vec3b(0, 0, 0); // default return black
}

cv::Vec3b PointCloud::getColorErrorRate(size_t const pt3D_idx) const
{
	cv::Vec3d Source(0, 255, 0);  // blue
	cv::Vec3d Target(255, 0, 0);  // red

	double minError = *std::min_element(mReprojectionErrors.begin(),mReprojectionErrors.end());
	double maxError = 10.0; //*std::max_element(mReprojectionErrors.begin(),mReprojectionErrors.end());

	double curError = mReprojectionErrors[pt3D_idx];

	if(curError > maxError) curError = maxError;

    int val = double(curError / (maxError - minError)) * 255;

    std::cout << "maxError:" << maxError << " minError:" << minError
    		<<  " error:" << mReprojectionErrors[pt3D_idx] << " val:" << val << std::endl;

	return cv::Vec3b(0, val, 255);
}

std::vector<int> PointCloud::getImgptForImg(size_t pt3D_idx) const {
	std::vector<int> ret(mCorresponding2D.size(),-1);
	int framenr = 0;

	for (auto i = mCorresponding2D.begin(); i != mCorresponding2D.end(); ++i) {
	    for (auto j = i->second.begin(); j < i->second.end(); ++j) {
			if(j->second == pt3D_idx) {
				//ret.push_back(j->first);
				ret[framenr] = j->first;
			}
		}
		framenr++;
	}

	return ret;
}


double PointCloud::getAverageError() const
{
  double sum = std::accumulate(mReprojectionErrors.begin(), mReprojectionErrors.end(), 0.0);
  return sum / mReprojectionErrors.size();
}

cv::Point3d PointCloud::getPoint3D(size_t i) const
{
  return cv::Point3d(mPoints[i](0)/mPoints[i](3), mPoints[i](1)/mPoints[i](3), mPoints[i](2)/mPoints[i](3));
}

std::vector<cv::Point3d> PointCloud::getPoints3D() const
{
  std::vector<cv::Point3d> pts;
  for (auto &p : mPoints) {
    pts.emplace_back(p(0)/p(3), p(1)/p(3), p(2)/p(3));
  }
  return pts;
}

std::vector<cv::Vec4d> PointCloud::getPoints() const
{
  return mPoints;
}

size_t PointCloud::size() const
{
  return mPoints.size();
}

void PointCloud::clearPoints()
{
  mPoints.clear();
  mReprojectionErrors.clear();
}

bool PointCloud::isViewable(size_t i) const
{
	bool ret = (!std::isnan(mPoints[i](0)) &&
	          !std::isnan(mPoints[i](1)) &&
	          !std::isnan(mPoints[i](2)) &&
	          (mPoints[i](2) > 0));

	return ret;
          //&& (mReprojectionErrors[i] <= REPROJECTION_ERROR_THRESHOLD));
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloud::createPclPointCloud()
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  size_t skipped_points = 0;
  //cv::Vec3b rgbv(0, 0, 0);
  //uint32_t rgb = ((uint32_t)rgbv(0) << 16 | (uint32_t)rgbv(1) << 8 | (uint32_t)rgbv(2));

  for (size_t i = 0; i < mPoints.size(); i++) {
    cv::Vec4d pt = mPoints[i];

    if (!isViewable(i) || (mReprojectionErrors[i] > 100)) {
      skipped_points++;
      continue;
    }

    cv::Vec3b rgbv = getColorOf3Dform2DPoint(i); //getColorErrorRate(i); //getColorOf3Dform2DPoint(i);
    //uint32_t rgb = ((uint32_t)rgbv(2) << 16 | (uint32_t)rgbv(1) << 8 | (uint32_t)rgbv(1));

    pcl::PointXYZRGB pclp((uint8_t)rgbv(2),(uint8_t)rgbv(1),(uint8_t)rgbv(0));
    pclp.x = pt(0) / pt(3);
    pclp.y = pt(1) / pt(3);
    pclp.z = pt(2) / pt(3);



    //pclp.rgb = *reinterpret_cast<float *>(&rgb);

    pcl_cloud->push_back(pclp);
  }

  std::cerr << "skipped " << skipped_points << "/" << mPoints.size() << " points for visualization" << std::endl;

  return pcl_cloud;
}

inline pcl::PointXYZRGB Eigen2PointXYZRGB(Eigen::Vector3f v, Eigen::Vector3f rgb) { pcl::PointXYZRGB p(rgb[0],rgb[1],rgb[2]); p.x = v[0]; p.y = v[1]; p.z = v[2]; return p; }

int	ipolygon[18] = {0,1,2, 0,3,1, 0,4,3, 0,2,4, 3,1,4, 2,4,1};

pcl::PolygonMesh visualizerGetCameraMesh(const Eigen::Matrix3f& R, const Eigen::Vector3f& _t,
		float r, float g, float b, double s = 0.01 /*downscale factor*/) {
	Eigen::Vector3f t = -R.transpose() * _t;

	Eigen::Vector3f vright = R.row(0).normalized() * s;
	Eigen::Vector3f vup = -R.row(1).normalized() * s;
	Eigen::Vector3f vforward = R.row(2).normalized() * s;

	Eigen::Vector3f rgb(r,g,b);

	pcl::PointCloud<pcl::PointXYZRGB> mesh_cld;
	mesh_cld.push_back(Eigen2PointXYZRGB(t,rgb));
	mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward + vright/2.0 + vup/2.0,rgb));
	mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward + vright/2.0 - vup/2.0,rgb));
	mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward - vright/2.0 + vup/2.0,rgb));
	mesh_cld.push_back(Eigen2PointXYZRGB(t + vforward - vright/2.0 - vup/2.0,rgb));

	pcl::PolygonMesh pm;
	pm.polygons.resize(6);
	for(int i=0;i<6;i++)
		for(int _v=0;_v<3;_v++)
			pm.polygons[i].vertices.push_back(ipolygon[i*3 + _v]);
	//pcl::toROSMsg(mesh_cld,pm.cloud);
	pcl::toPCLPointCloud2(mesh_cld,pm.cloud);
	return pm;
}

pcl::PolygonMesh visualizerGetCameraMesh(const cv::Matx33f& R, const cv::Vec3f& t, float r, float g, float b, double s) {
	return visualizerGetCameraMesh(Eigen::Matrix<float,3,3,Eigen::RowMajor>(R.val),Eigen::Vector3f(t.val),r,g,b,s);
}

/*
void PointCloud::keyboardcb(const cv::viz::KeyboardEvent& ke ) {

}
*/

void PointCloud::RunVisualization(std::string name)
{
	//std::cout << "creating viewer" << std::endl;

	pcl::visualization::PCLVisualizer viewer(name);
	//viewer.registerKeyboardCallback(keyboardcb());

	viewer.setBackgroundColor(255, 255, 255);
	//viewer.setBackgroundColor(0, 0, 0);

	// add camera
	int j=0;
	float camera_color[] = {255.0,0.0,0.0};
	double camera_skale_faktor = 1.0;

	for (auto i = mCorresponding2D.begin(); i != mCorresponding2D.end(); ++i) {
		cv::Matx34d v = i->first->getProjectionMatrix();

		//std::cout << i->first->getProjectionMatrix() << std::endl << std::endl;

		std::stringstream ss; ss << "camera" << j; j++;
		cv::Matx33f R;
		R(0,0)=v(0,0); R(0,1)=v(0,1); R(0,2)=v(0,2);
		R(1,0)=v(1,0); R(1,1)=v(1,1); R(1,2)=v(1,2);
		R(2,0)=v(2,0); R(2,1)=v(2,1); R(2,2)=v(2,2);

		cv::Vec3f t(v(0,3),v(1,3),v(2,3));
		pcl::PolygonMesh cam_mesh = visualizerGetCameraMesh(R,t,
				camera_color[0],camera_color[1],camera_color[2],
				camera_skale_faktor);

		viewer.addPolygonMesh(cam_mesh,ss.str());
	}

	//viewer.removePolygonMesh("camera0");

	//std::cout << "populating pointcloud" << std::endl;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud = createPclPointCloud();

	//std::cout << "adding pointcloud" << std::endl;
	viewer.addPointCloud(pcl_cloud);
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5);

	/*
    viewer.setCameraPosition(
    		1.8291,  0.2093, -19.9923,
    		0.1596, -0.9861,   -0.0450,
            0.0,    -1.0,       0.0);
    */
	viewer.setCameraPosition(
			 -15.5240, -14.7910, -37.4616,
			  -4.1836,  -8.3018,   5.6482,
			  -0.0555,  -0.9851,   0.1629);



    std::vector<pcl::visualization::Camera> cam;

	std::cout << "starting viewer" << std::endl;
	while (!viewer.wasStopped()) {
		viewer.spinOnce(100);
		std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));


		cam.clear();
		viewer.getCameras(cam);
		/*
		cout << "Cam: (" << endl
			 << cam[0].pos[0]   << ", " << cam[0].pos[1]   << ", " << cam[0].pos[2]   << "," << endl  // position
			 << cam[0].focal[0] << ", " << cam[0].focal[1] << ", " << cam[0].focal[2] << "," << endl  // focal
			 << cam[0].view[0]  << ", " << cam[0].view[1]  << ", " << cam[0].view[2]  << ");" << endl; // view
		*/


	}
}

bool PointCloud::UpdatePoints(std::vector<cv::Vec4d> newPtCl) {
	if(newPtCl.size() == mPoints.size()) {
		mPoints = newPtCl;
		return true;
	} else {
		std::cout << "new:" << newPtCl.size() << " old:" << mPoints.size() << std::endl;
		return false;
	}

}


