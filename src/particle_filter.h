#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <string>
#include <tuple>

#include "opencv2/opencv.hpp"

struct Particle {
	std::set<std::pair<int, int>> pix;
	std::pair<int, int> last;
	bool complete1 = false;
};

struct Bbox {
	int x_start;
	int y_start;
	int x_end;
	int y_end;
	
	Bbox(int x1, int x2, int y1, int y2) {
		x_start = x1;
		x_end = x2; 
		y_start = y1;
		y_end = y2;
	}
};

/*
struct ImageD : public cv::Mat<cv::CV_64FC1> {
	// std::vector<std::vector<double>> pixels;
	// int x_size, y_size;
	Bbox area;

	ImageD(Mat& img) {
		area = Bbox(0, x_size, 0, y_size);
	}
};*/

class ParticleFilter {
public:
	cv::Mat d_img;
	cv::Mat d_denoised_img;
	int d_threshold;
	cv::Mat d_grad_x;
	cv::Mat d_grad_y;
	double d_variance;

	int Threshold();
	int LoadImage(const std::string& file);
	void Clear();
	//void Prepare();
	void Denoise();

	void FindGradients();
};