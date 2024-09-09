#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	double a[3][3] = {
		{1, 2, 3}, 
		{3, 5, 7},
		{8, 6, 4}
	};

	cv::Mat m = cv::Mat(3, 3, CV_64F, &a);

	cout << "a = " << m << endl;
	return 0;
}