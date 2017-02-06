/*
 * VideoStream.h
 *
 *  Created on: Sep 9, 2016
 *      Author: josef
 */

#ifndef VIDEOSTREAM_H_
#define VIDEOSTREAM_H_

#include <opencv2/core.hpp>
#include <Frame.h>
#include <Camera.h>

class IVideoStream {
public:
	IVideoStream() {};
	virtual ~IVideoStream() {};

	virtual bool isOpened() const = 0;
	virtual int width() const = 0;
	virtual int height() const = 0;

	//virtual void getFrame(cv::Mat &frame) = 0;
	virtual Frame * getFrame() = 0;
};

#endif /* VIDEOSTREAM_H_ */
