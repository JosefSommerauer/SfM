#include "LiveStream.h"

#include <cassert>
#include <iostream>

#include <unistd.h>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "StopWatch.h"
#include "IVideoStream.h"


LiveStream::LiveStream(int camNum, Camera * cam) : mCamera(camNum), mCameraOptions(cam)
{
  if (!openCamera(camNum, cam->getWidth(), cam->getHeight())) {
    return;
  }
}

LiveStream::~LiveStream()
{
	mCamera.release();
}

#define FOURCC(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))

std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer, 128, pipe.get()) != NULL)
            result += buffer;
    }
    return result;
}

bool LiveStream::openCamera(int num, int width, int height)
{
  //assert(num >= 0);
  //mCamera.open(num);

  if (!mCamera.isOpened()) {
    std::cerr << "could not open camera" << std::endl;
    return false;
  }

  /*
  double codec = FOURCC('M', 'J', 'P', 'G');
  if (!mCamera.set(cv::CAP_PROP_FOURCC, codec))
  //if (!mCamera.set(cv::CAP_PROP_FOURCC, FOURCC('M', 'J', 'P', 'G')))
  {
    char *fourcc = (char *) &codec;
    std::cerr << "could not set codec " << fourcc[0] << fourcc[1] << fourcc[2] << fourcc[3] << std::endl;
  }
  */

  if ((width != -1) && (height != -1)) {
    // both calls will return false
    mCamera.set(cv::CAP_PROP_FRAME_WIDTH, width);
    mCamera.set(cv::CAP_PROP_FRAME_HEIGHT, height);
  }

  mStreamWidth = mCamera.get(cv::CAP_PROP_FRAME_WIDTH);
  mStreamHeight = mCamera.get(cv::CAP_PROP_FRAME_HEIGHT);

  if (   ((width != -1) && (mStreamWidth != width))
      || ((height != -1) && (mStreamHeight != height))) {
    std::cerr << "could not set resolution of camera " << num << " to: " << width << "x" << height
              << " current resolution is: " << mStreamHeight << "x" << mStreamWidth << std::endl;
    mCamera.release();
    return false;
  }


  std::vector<std::pair<std::string,int> > uvcdyn_controlls = {
		  {"Focus, Auto",                    0},
		  {"Focus (absolute)",               0},
		  {"Exposure, Auto Priority",        0},
		  {"Exposure, Auto",                 0},
		  {"Exposure (Absolute)",          300},
		  {"White Balance Temperature, Auto",0},
		  {"White Balance Temperature",   4000},
		  {"Brightness",                   150},
		  {"Contrast",                     150},
		  {"Saturation",                   200},
		  {"Sharpness",                    100}
  };

  std::stringstream ss, res;
  for (int i = 0; i < uvcdyn_controlls.size(); ++i) {
	  ss << "uvcdynctrl -v -d video" << num <<" -s '"
	     << uvcdyn_controlls[i].first << "' "
	     << uvcdyn_controlls[i].second << std::endl;
  }

  exec(ss.str().c_str()); ss.str(std::string());

  std::cout << "initialized camera " << num << " with "
            << mStreamWidth << "x" << mStreamHeight << std::endl;
  return true;
}

bool LiveStream::isOpened() const
{
  return mCamera.isOpened();
}

int LiveStream::width() const
{
  return mStreamWidth;
}

int LiveStream::height() const
{
  return mStreamHeight;
}

Frame * LiveStream::getFrame() {
	bool breakwhile = false;
	bool quit = false;

	cv::Mat img_tmp, img_tmp_a;
	Frame * currentFrame = 0;

	cv::namedWindow("Live", cv::WINDOW_NORMAL);
	cv::resizeWindow("Live", 640, 480);
	cv::moveWindow("Live", 0, 0);

	if(mLastFrame != 0) {
		cv::namedWindow("Previous", cv::WINDOW_NORMAL);
		cv::resizeWindow("Previous", 640, 480);
		cv::moveWindow("Previous", 640, 0);
		cv::imshow("Previous", mLastFrame->getColorImage());
	}

	do {
		mCamera >> img_tmp_a;
		cv::undistort(img_tmp_a, img_tmp, mCameraOptions->getCalibrationMatrix(), mCameraOptions->getDistortionCoefficients());

		cv::imshow("Live", img_tmp);

		char key = cv::waitKey(30);

		switch (key) {
		case 'c': // grap image
		{
			currentFrame = new Frame(img_tmp);

			if (currentFrame != 0) breakwhile = true;

			break;
		}
		case 'q': {
			mCamera.release();

			return 0;
		}
		}
	} while (!breakwhile);

	mLastFrame = currentFrame;

	cv::destroyWindow("Live");
	cv::destroyWindow("Previous");

	return currentFrame;
}


