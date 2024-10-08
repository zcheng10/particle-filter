#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <tuple>
#include <cmath>
#include "opencv2/opencv.hpp"
#include <algorithm>

#include "spdlog/spdlog.h"



struct Particle {
	std::set<std::pair<int, int>> pix;
	
	bool complete1 = false;
	bool complete2 = true;

	int dirpenum1;
	int dirpenum2;

	double weight;

	std::pair<int, int> last1;
	std::pair<int, int> last2;
	std::pair<int, int> penum1;
	std::pair<int, int> penum2;

	/** You must have an empty constructor, otherwise `std::vector<Particle>`
	    can not work
	 */
	Particle() = default;

	Particle(std::pair<int, int> last, std::pair<int, int> penum, 
		int dir, double w) {
		if (dir == 1) {
			last1 = last;
			penum1 = penum;
			last2 = penum;
			penum2 = last;
		}
		else {
			last1 = penum;
			penum1 = last;
			last2 = last;
			penum2 = penum;
		}

		pix.insert(last);
		pix.insert(penum);
		complete1 = false;
		complete2 = true;
		weight = w;
	}

	void print() {
		std::stringstream os;
		// os << "Particle: " << std::endl;
		const std::string SEP = "; ";
		os << "Number of Pixels: " << pix.size() << SEP;
		os << "Weight = " << weight << std::endl;
		os << "Last1 = " << last1.first << ", " << 
			last1.second << SEP;
		os << "Last2 = " << last2.first << ", " << 
			last2.second << std::endl;
		os << "Penum1 = " << penum1.first << ", " <<
			penum1.second << SEP;
		os << "Penum2 = " << penum2.first << ", " << 
			penum2.second << SEP;
		os << "Complete1 = " << complete1 << ", Complete2 = " << complete2 << std::endl;
		spdlog::info("Particle: {}", os.str());
	}
};

struct Bbox {
	int x_start;
	int y_start;
	int x_end;
	int y_end;
	
	Bbox() = default;

	Bbox(int x1, int x2, int y1, int y2) {
		x_start = x1;
		x_end = x2; 
		y_start = y1;
		y_end = y2;
	}

	void shrinkBy(int x) {
		x_start += x;
		x_end -= x;
		y_start += x;
		y_end -= x;
	}

	void print() {
		std::cout << x_start << ", " << x_end << std::endl;
		std::cout << y_start << ", " << y_end << std::endl;
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

struct WeightTuple {
	double weight;
	int index;
	std::pair<int, int> next;
	int dirpenum;
	int dir;

	bool operator < (const WeightTuple& w) const {
		const bool equalWeight = fabs(weight - w.weight) < 1e-10;
		if (equalWeight && index == w.index 
			&& next == w.next && dirpenum == w.dirpenum) {
			return dir < w.dir;
		}

		if (equalWeight && index == w.index && next == w.next) {
			return dirpenum < w.dirpenum;
		}

		if (equalWeight && index == w.index) {
			return next < w.next;
		}

		if (equalWeight) {
			return index < w.index;
		}

		return weight < w.weight;
	}

	WeightTuple(double w, int ind, int nex1, int nex2, int dpen, int d) {
		weight = w;
		index = ind;
		next.first = nex1;
		next.second = nex2;
		dirpenum = dpen;
		dir = d;
	}
};

class ParticleFilter {
public:
	double d_threshold = 0.14;
	double d_zoom_scale = 4.0;
	double d_zoom_weight = 0.2;
	double d_lambda = 150;
	int d_N = 4;

	int d_I0;
	double d_variance;
	int d_rows;
	int d_cols;

	Bbox d_area;

	cv::Mat d_img;
	cv::Mat d_denoised_img;
	cv::Mat d_grad_x;
	cv::Mat d_grad_y;
	cv::Mat d_constraints;
	cv::Mat d_logconstraints;

	std::vector<Particle> d_particles;
	std::vector<Particle> d_contourpaths;
	std::vector<std::vector<bool>> d_found;

	ParticleFilter(double thres, double l, double sc, double we) {
		d_threshold = thres;
		d_lambda = l;
		d_zoom_scale = sc;
		d_zoom_weight = we;
	}

	void Convolve(cv::Mat& m, cv::Mat& res, std::vector<std::vector<double> > & kern);
	void Threshold();
	int LoadImage(const std::string& file);
	void Clear();
	void Prepare();
	void Denoise();
	void FindConstraints();
	bool FindStartpoint(std::pair<int, int>& startpoint);
	bool FindNext();
	bool FindNextDir(int i, int dir, std::vector<WeightTuple>& weights);
	void FindCandidates(std::vector<WeightTuple>& weights);
	int FindFinalParticle();
	void FindGradients();
	void MakeGradientsContinuous(int& graddir, int& prevgraddir);
	int FindVectorDirection(double x, double y);
	void FindContours(const std::string filename);
	bool AtEdge(int x, int y);
	bool PixValid(int x, int y);
	void InitializeParticles(std::pair<int, int>& startpoint);
	std::vector<Particle> GetContourPaths();
};

/** get env variable */
template <typename T>
void GetEnv(const std::string& VAR, T& t)
{
	const char* ev = std::getenv(VAR.c_str());
	if (ev) {
		std::istringstream is(ev);
		is >> t;
	}
}