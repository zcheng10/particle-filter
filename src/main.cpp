#include "particle_filter.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	/*double a[3][3] = {
		{1, 2, 3}, 
		{3, 5, 7},
		{8, 6, 4}
	};

	cv::Mat m = cv::Mat(3, 3, CV_64F, &a);

	cout << "a = " << m << endl;*/

	const string file_name = "Image1.jpg";
	ParticleFilter filter = (0.14, 150, 4.0, 0.2);
	filter.FindContours(file_name);

	const string output_file = "Output_" + file_name;
	cv::Mat Img = filter.d_img;
	cv::Mat outImg;
	cv::cvtColor(Img, outImg, cv::COLOR_GRAY2BGR);
	for(auto& it : filter.d_contourpaths){
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