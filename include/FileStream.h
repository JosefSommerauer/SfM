/*
 * FileStream.h
 *
 *  Created on: Sep 10, 2016
 *      Author: josef
 */

#ifndef FILESTREAM_H_
#define FILESTREAM_H_

#include <vector>
#include <opencv2/core.hpp>
#include "IVideoStream.h"
#include "Frame.h"

class FileStream: public IVideoStream {
public:
	FileStream();

	FileStream(std::string const & filename, Camera * cam);

	virtual ~FileStream();

	bool isOpened() const;
	int width() const;
	int height() const;

	//void getFrame(cv::Mat &frame);

	Frame * getFrame();
private:
	int mStreamWidth = 0;
	int mStreamHeight = 0;
	int mCurrentFrame = 0;

	std::vector<Frame> mFrames;
	Camera * mCameraOptions;
};

#endif /* FILESTREAM_H_ */
