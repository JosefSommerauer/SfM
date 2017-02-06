/*
 * FileStream.cpp
 *
 *  Created on: Sep 10, 2016
 *      Author: josef
 */

#include <string>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "FileStream.h"

FileStream::FileStream() {
	// TODO Auto-generated constructor stub

}

FileStream::~FileStream() {
	// TODO Auto-generated destructor stub
}

FileStream::FileStream(std::string const & _filename, Camera * cam) {
  mCameraOptions = cam;
  int cnt=0;

  std::ifstream myfile(_filename.c_str());
  if (!myfile.is_open()) {
	std::cout << "Unable to read file: " << _filename << std::endl;
  } else {;
	size_t found = _filename.find_last_of("/\\");
	std::string line_str;
	std::string path_to_file = (found!=std::string::npos)?_filename.substr(0, found) + std::string("/"):"";
	while ( getline(myfile, line_str) )  {
		if(line_str[0] != '#' && line_str.length() > 3) {
			std::string filename(path_to_file+line_str);

			cv::Mat img = cv::imread(filename);
			cv::Mat img_undistored;

			if(img.empty()) {
				std::cout << "error: unable to read image " << filename << std::endl;
			} else {
				cv::undistort(img, img_undistored, mCameraOptions->getCalibrationMatrix(), mCameraOptions->getDistortionCoefficients());
				img = img_undistored;

				if(mStreamWidth == 0 || mStreamHeight == 0) {
					mStreamWidth = img.cols;
					mStreamHeight = img.rows;
				}

				if(img.cols != mStreamWidth && img.rows != mStreamHeight) {
					std::cout << "error: images dont have the same size. skip image " << filename << std::endl;
				} else {
					mFrames.push_back(Frame(img));
				}
			}
		}
		cnt++;
	}

	std::cout << mFrames.size() << " images have been read. "
			  << cnt-mFrames.size() << " had been skipped." << std::endl
			  << std::string(80,'-') << std::endl;

  }
}

bool FileStream::isOpened() const {
	if(mFrames.size() > 0 && mCurrentFrame < mFrames.size()) {
		return true;
	} else {
		return false;
	}
}

int FileStream::width() const {
	return mStreamWidth;
}

int FileStream::height() const {
	return mStreamWidth;
}

Frame * FileStream::getFrame() {
	if(isOpened()) {
		return &mFrames[mCurrentFrame++];
	} else {
		return 0;
	}
}
