#ifndef LIVESTREAM_H_INCLUDED
#define LIVESTREAM_H_INCLUDED

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <mutex>
#include <thread>

#include "IVideoStream.h"
#include "Frame.h"

class LiveStream : public IVideoStream {

private:
  cv::VideoCapture mCamera;
  int mStreamWidth = 0;
  int mStreamHeight = 0;
  Frame * mLastFrame = 0;
  Camera * mCameraOptions;

  bool openCamera(int num, int width, int height);

public:
  LiveStream(int camNum, Camera * cam);

  virtual ~LiveStream();

  bool isOpened() const;
  int width() const;
  int height() const;

  Frame * getFrame();
};

#endif
