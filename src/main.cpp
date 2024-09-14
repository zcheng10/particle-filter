#include "particle_filter.h"
#include <tuple>

using namespace std;
using namespace cv;

/** a simple test of whether OpenCV is installed correctly
 */
void test_cv() {
	double a[3][3] = {
		{1, 2, 3},
		{3, 5, 7},
		{8, 6, 4}
	};

	cv::Mat m = cv::Mat(3, 3, CV_64F, &a);

	cout << "a = " << m << endl;
}

/** split a file name into base and ext 
*/
std::pair<std::string, std::string> splitext(const std::string& name)
{
	size_t p = name.rfind(".");
	if (p == string::npos)
		return std::make_pair(name, "");
	return std::make_pair(name.substr(0, p), name.substr(p));
}

int main(int argc, char** argv) {
	if (argc < 2) {
		// -- print usage
		cout << "Usage: " << string(argv[0]) << "  [image_file]" << endl;
		return 0;
	}

	const string file_name = std::string(argv[1]);
		// "Image1.jpg";
	ParticleFilter filter = (0.14, 150, 4.0, 0.2);
	filter.FindContours(file_name);
	cout << "contour extracted" << endl;

	std::string baseName, ext;
	std::tie(baseName, ext) = splitext(file_name);
	const string output_file = baseName + "_output" + ext;
	cout << "output file = " << output_file << endl;
	
	cv::Mat Img = filter.d_img;
	cv::Mat outImg;
	cv::cvtColor(Img, outImg, cv::COLOR_GRAY2BGR);
	for (auto& it : filter.d_contourpaths) {
		for (auto& it2 : it.pix) {
			cv::Vec3b pixch = outImg.at<double>(it2.first, it2.second);
			pixch[0] = 0.0;
			pixch[1] = 0.0;
			pixch[2] = 255.0;
		}
	}

	cv::imwrite(output_file, outImg);
	return 0;
}