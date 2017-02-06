#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <map>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iomanip>      // std::setw

#include "Frame.h"
#include "IterativeLinearLS.h"
#include "GPUIterativeLinearLS.h"
#include "GPUSurf.h"
#include "IVideoStream.h"
#include "LiveStream.h"
#include "FileStream.h"
#include "OpenCVFeatureDetector.h"
#include "OpenCVPoseEstimation.h"
#include "options.h"
#include "RubikManualFeatures.h"
#include "ViewCombination.h"
#include "OpticalFlowFeatures.h"
#include "Camera.h"
#include "StopWatch.h"
#include "FileWriter.h"
#include "tests.h"

#include "SfM.h"

#include "cuda/triangulation.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;



std::ostream &operator<<(std::ostream &out, SfMTimes const &times) {
    out.setf(std::ios::fixed,std::ios::floatfield);
    out.precision(4);

	out << "Feature Detect: \t" << std::setw(10) << times.feature_detection << " ms" << std::endl
		<< "Pose Estimation:\t" << std::setw(10) << times.pose_estimation << " ms" << std::endl
		<< "Triangulation:  \t" << std::setw(10) << times.triangulation << " ms" << std::endl
		<< "SfM took:       \t" << std::setw(10)
	    << times.feature_detection + +times.pose_estimation
					+ +times.triangulation << " ms";
	return out;
}

int main(int argc, char **argv) {
	// get configuration
	Options opts;
	int nopts = check_options(opts, argc, argv);
	if (nopts == -1) {
		usage(argv[0], opts);
		return 1;
	}
	argc -= nopts;	argv += nopts;

	cout << "Options:" << endl << opts << endl;

	std::string calib_file = opts.camera_calib_file;

	int width = 0;
	int height = 0;
	cv::Mat K, kappa;

	if (!calib_file.empty()) {
		cv::FileStorage fs(calib_file, cv::FileStorage::READ); // Read the settings

		if (!fs.isOpened()) {
			cout << "could not open the camera calibration file: \""
					<< calib_file << "\"" << endl;
			return -1;
		}

		width = fs["image_width"];
		height = fs["image_height"];
		fs["camera_matrix"] >> K;
		fs["distortion_coefficients"] >> kappa;
		kappa = kappa.t();
	}

	// initialize objects
	Frame *frame_last, *frame_now;

	IVideoStream * stream = 0;
	Camera * cam = 0;
	int img_height = (height != 0) ? height : opts.height;
	int img_width  = (width != 0) ? width : opts.width;

	std::cout << std::string(80,'-') << std::endl;
	if (!K.empty() && !kappa.empty()) {
		cam = new Camera(img_width, img_height, K, kappa);
	} else {
		cam = new Camera(img_width, img_height);
	}

	std::cout << "Calibration Matrix:" << std::endl << cam->getCalibrationMatrix() << std::endl << std::endl;
	std::cout << "Distortion Coefficients: " << cam->getDistortionCoefficients() << std::endl;
	std::cout << std::string(80,'-') << std::endl;

	IFeatureDetect *fdetect = JS::getFeatureDetector(opts.feat_detector, opts.use_cross_matching);

	if(fdetect == nullptr) {
		std::cout << opts.feat_detector << " is not a valid detector!" << std::endl;
		return -1;
	}

	ITriangulation *triang = JS::getTriangulation(opts.triangulation);
	IPoseEstimation *poseestim = new OpenCVPoseEstimation(triang);

	assert(fdetect != NULL && triang != NULL && poseestim != NULL);

	// load images
	if (opts.live_stream) {
		stream = new LiveStream(opts.cam_num, cam);
	} else {
		std::string imgfile = opts.image_list_file.value;
		stream = new FileStream(imgfile, cam);
	}

	if (stream == NULL || !stream->isOpened()) {
		if(opts.live_stream) {
			std::cerr << "could not open camera "
					  << opts.cam_num.value << std::endl;
		}else {
			std::cerr << "image list file " << opts.image_list_file.value
					  << " is not valid" << std::endl;
		}

		return -1;
	}




	/*
	if (opts.test_features) {
		test_features(stream, opts.live_stream, img1, img2);
		return 0;
	}

	if (opts.test_triangulation) {
		Frame frame_last(img_last);
		Frame frame_now(img_now);
		Camera cam(img_now.cols, img_now.rows);
		testGPUTriangulation(cam, &frame_last, &frame_now);
		return 0;
	}
	*/

	/*
	if (opts.feat_detector.value == "RUB") {
		RubikManualFeatures *rub = dynamic_cast<RubikManualFeatures *>(fdetect);
		rub->preloadFrame(frame_now, 1);
	}
	*/

	//size_t rubik_counter = 2;

	PointCloud global_points;

	SfMTimes times_initial = { 0 };
	std::vector<SfMTimes> times_average;
	size_t times_counter = 0;

	JS::SfM mSfM(fdetect,poseestim,triang,*cam);

	frame_now = stream->getFrame();
	fdetect->findFeatures(frame_now);

	bool isPoseEstimated = false;
	int cnt=0;
	do {
		frame_last = frame_now;
		frame_now = stream->getFrame();
		if(frame_now == NULL) break;

		/*
		if (opts.feat_detector.value == "RUB") {
			if (rubik_counter > 4) {
				std::cout << "enough  rubik iterations" << std::endl;
				break;
			}
			std::cout << "=== Processing RUBIK features #" << rubik_counter
					<< " ===" << std::endl;
			RubikManualFeatures *rub =
					dynamic_cast<RubikManualFeatures *>(fdetect);
			rub->preloadFrame(frame_now, rubik_counter++);
		}
		*/


		times_average.emplace_back();

		std::cout << "processing frame " << cnt++  << std::endl;
		isPoseEstimated = mSfM.isPoseEstimated();
		if (mSfM.Process(frame_last,frame_now, times_average[times_counter],
				opts.export_keypoints, opts.show_matches, opts.show_points, opts.use_bundle_adjuster)) {

			cout << std::string(80,'-') << std::endl;
			cout << "SFM TOOK: " << std::endl << times_average[times_counter] << std::endl
				 << std::string(80,'-') << std::endl;
			if (!isPoseEstimated) {
				times_initial = times_average[times_counter];
				times_average.pop_back();
			} else {
				times_counter++;
			}
		}

	} while (stream->isOpened() || (opts.feat_detector.value == "RUB"));

	// calculate times and cleanup
	if(mSfM.GetPointCloud().size() > 0) {
		mSfM.PrintMatchMatrix();

		if(times_average.size() > 0) {
			SfMTimes avg_times;
			avg_times.feature_detection = 0;
			avg_times.pose_estimation = 0;
			avg_times.triangulation = 0;

			for_each(times_average.begin(), times_average.end(),
					[&avg_times](SfMTimes const &t)
					{
						avg_times.feature_detection += t.feature_detection;
						avg_times.pose_estimation += t.pose_estimation;
						avg_times.triangulation += t.triangulation;
					});

			avg_times.feature_detection /= times_average.size();
			avg_times.pose_estimation /= times_average.size();
			avg_times.triangulation /= times_average.size();

			cout << "------ SfM run " << times_counter + 1 << " times " << std::string(80-23,'-') << endl;
			cout << "INITIAL: " << std::endl << times_initial << endl;
			cout << "AVERAGE: " << std::endl << avg_times << endl;
			cout << std::string(80,'-') << endl;
		}

		cout << std::string(80,'-') << endl;
		cout << "Recovered Object Points:\t" << mSfM.GetPointCloud().size() << endl;

		std::cout << "Starting visualization" << std::endl;
		//global_points.RunVisualization("Global points");
		mSfM.RunVisualization("Global points");
	}




	delete cam;
	delete stream;
	delete fdetect;
	delete poseestim;
	delete triang;


	return 0;
}
