#ifndef OPTIONS_H_INCLUDED
#define OPTIONS_H_INCLUDED

#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <cstdlib>



template <typename T>
class Option {
public:
  Option(T default_val, std::string s_opt, std::string l_opt, std::string name, std::string desc)
        : value(default_val), short_option(s_opt), long_option(l_opt),
          name(name), description(desc) { }
  T value;
  std::string short_option;
  std::string long_option;
  std::string name;
  std::string description;

  static bool has_arg;

  bool process(std::string const &opt, int &argc, char const * const *argv)
  {
    if (   ((short_option != "") && (opt == short_option))
        || (opt == long_option)) {
      argc++;
      if (has_arg) {
        if (argv[argc] == NULL) {
            std::cerr << "missing value for " << opt << std::endl;
            return false;
        }
        set(argv[argc]);
        argc++;
      } else {
        enable();
      }
      return true;
    }
    return false;
  }

  void set(char const *arg)
  {
    std::cerr << "missing specialization for option " << name << std::endl;
    assert(false);
  }

  void enable()
  {
    std::cerr << "missing specialization for enable in option " << name << std::endl;
    assert(false);
  }

  std::string str() const
  {
    std::stringstream str;
    str << name << ": " << std::boolalpha << value;
    return str.str();
  }

  operator T() const
  {
    return value;
  }
};

template <typename T>
bool Option<T>::has_arg = true;

template <>
bool Option<bool>::has_arg = false;

template <>
void Option<bool>::enable()
{
  value = true;
}

template <>
void Option<std::string>::set(char const *arg)
{
  value = std::string(arg);
}

template <>
void Option<int>::set(char const *arg)
{
  value = atoi(arg);
}

template <typename T>
std::ostream &operator<<(std::ostream &out, Option<T> const &o)
{
  out << " " << o.short_option
      << ", " << o.long_option
      << ": " << o.description;
  return out;
}

class Options {
public:
  Option<int> cam_num = Option<int>(0, "-c", "--camera", "Camera",
                                    "Number of the camera to capture. E.g. 0 for /dev/video0");
  Option<int> width = Option<int>(1280, "-w", "--width", "Width",
                                  "Width of the captured image");
  Option<int> height = Option<int>(720, "-h", "--height", "Height",
                                  "Height of the captured image");
  Option<bool> test_features = Option<bool>(false, "", "--test-features", "Test Features",
                                           "Test Feature Detector/Extractor/Matcher");
  Option<bool> test_triangulation = Option<bool>(false, "", "--test-triangulation", "Test Triangulation",
                                                 "Test GPU Triangulation");
  Option<bool> show_matches = Option<bool>(false, "-m", "--show-matches", "Show Matches",
                                           "Visualize the matches found by FetureDetection");
  Option<std::string> feat_detector = Option<std::string>("SURF", "-f", "--feature-detector", "Feature Detector",
                                                   "Choose the feature detector to use: OF, GPU (SURF), ORB, ORB5000, SURF, SIFT, AKAZE, RUB");
  Option<bool> live_stream = Option<bool>(false, "-l", "--live", "Live Stream",
                                          "Switch on livestream from camera");
  Option<bool> export_keypoints = Option<bool>(false, "-e", "--export_keypoints", "Export Keypoints",
                                               "Export Keypoints after detection");
  Option<std::string> triangulation = Option<std::string>("OCV", "-t", "--triangulation", "Triangulation Method",
                                                          "Set the triangulation method: OCV or GPU");
  Option<bool> show_points = Option<bool>(false, "-p", "--show-points", "Show Points",
                                          "Visualizes the triangulated points during sfm");

  Option<std::string> camera_calib_file = Option<std::string>("", "-k", "--calibration", "Camera Calibration File",
                                          "Load Camera Calibration form file. to gernate this file use opencv camera calibration.");

  Option<std::string> image_list_file = Option<std::string>("images.txt", "-i", "--imagelist", "Imagelist File",
                                          "Load images from files. file contains list of pathes to images separated by linefeed.");

  Option<bool>use_bundle_adjuster = Option<bool>(false, "-ba", "--bundle_adjuster", "Use Bundle Adjuster",
          	  	  	  	  	  	  	  	  	  	  "Use Bundle Adjuster");
  Option<bool>use_cross_matching = Option<bool>(false, "-xm", "--cross_matching", "Use Cross Matching",
          	  	  	  	  	  	  	  	  	  	  "Use Bundle Adjuster. If not set Lowe test is used.");


  std::string print_all() const
  {
    std::stringstream ss;
    ss              << cam_num
       << std::endl << width
       << std::endl << height
       << std::endl << test_features
       << std::endl << test_triangulation
       << std::endl << show_matches
       << std::endl << feat_detector
       << std::endl << live_stream
       << std::endl << export_keypoints
       << std::endl << triangulation
       << std::endl << show_points
       << std::endl << camera_calib_file
       << std::endl << image_list_file
       << std::endl << use_bundle_adjuster
       << std::endl << use_cross_matching
       << std::endl;
    return ss.str();
  }

  bool process(std::string const &opt, int &argc, char const * const *argv) {
    std::cout << "processing: " << argv[argc] << std::endl;
    if (   !cam_num.process(opt, argc, argv)
        && !width.process(opt, argc, argv)
        && !height.process(opt, argc, argv)
        && !test_features.process(opt, argc, argv)
        && !test_triangulation.process(opt, argc, argv)
        && !show_matches.process(opt, argc, argv)
        && !feat_detector.process(opt, argc, argv)
        && !live_stream.process(opt, argc, argv)
        && !export_keypoints.process(opt, argc, argv)
        && !triangulation.process(opt, argc, argv)
        && !show_points.process(opt, argc, argv)
        && !camera_calib_file.process(opt, argc, argv)
        && !image_list_file.process(opt, argc, argv)
        && !use_bundle_adjuster.process(opt, argc, argv)
        && !use_cross_matching.process(opt, argc, argv)
       )
    {
      std::cerr << "unknown option " << opt << std::endl;
      return false;
    }

    return true;
  }
};

inline std::ostream &operator<<(std::ostream &out, Options const &o)
{
  //std::string tmp = o.print_all();

  out << "-- " << o.cam_num.str()             << std::endl
	  << "-- " << o.width.str()               << std::endl
	  << "-- " << o.height.str()              << std::endl
	  << "-- " << o.test_features.str()       << std::endl
	  << "-- " << o.test_triangulation.str()  << std::endl
	  << "-- " << o.show_matches.str()        << std::endl
	  << "-- " << o.feat_detector.str()       << std::endl
	  << "-- " << o.live_stream.str()         << std::endl
	  << "-- " << o.export_keypoints.str()    << std::endl
	  << "-- " << o.triangulation.str()       << std::endl
	  << "-- " << o.show_points.str()         << std::endl
	  << "-- " << o.camera_calib_file.str()   << std::endl
	  << "-- " << o.image_list_file.str()     << std::endl
	  << "-- " << o.use_bundle_adjuster.str() << std::endl
	  << "-- " << o.use_cross_matching.str()  << std::endl;

  return out;
}

static void usage(char const * const progname, Options const &o) {
	std::cout << "usage:" << std::endl << progname << " [OPTIONS]" << std::endl << std::endl
			<< "Options:" << std::endl << o.print_all() << std::endl << std::endl;
}

// returns processed arguments
int check_options(Options &opts, int const argc, char const * const *argv) {
	int i = 1;
	while (i < argc) {
		std::string arg(argv[i]);

		if (arg[0] != '-') {
			break;
		}

		if (arg == "--help") {
			return -1;
		}

		if (!opts.process(arg, i, argv)) {
			return -1;
		}
	}

	return i;
}

#endif
