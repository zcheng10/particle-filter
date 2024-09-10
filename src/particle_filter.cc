#include <particle_filter.h>

using namespace std;

int ParticleFilter::Threshold() {
	int xi = d_img.area.x_start;
	int xf = d_img.area.x_end;
	int yi = d_img.area.y_start;
	int yf = d_img.area.y_end;
	
	const int L = 256;
	vector<int> gray_levels(L, 0);
	for (int i = xi; i < xf; i++) {
		for (int j = yi; j < yf; j++) {
			int ind = (int)d_denoised_img.at(i, j);
			gray_levels[ind]++;
		}
	}

	vector<int> gray_levels_cum(L, 0);
	vector<double> weighted_gray_levels_cum(L, 0.0);
	double mean = 0.0;
	int tot = d_denoised_img.row * d_denoised_img.col;
	gray_levels_cum[0] = gray_levels[0];
	int I0 = 0;
	double mx = 0.0;
	for (int i = 1; i < L; i++) {
		gray_levels_cum[i] += gray_levels_cum[i - 1];
		gray_levels_cum[i] += gray_levels[i];
		weighted_gray_levels_cum[i] += weighted_gray_levels_cum[i - 1];
		weighted_gray_levels_cum[i] += 1.0 * i * gray_levels[i];
		mean += i * 1.0 * gray_levels[i] / tot;
	}

	double weighted_tot = weighted_gray_levels_cum.back();
	for (int i = 0; i < L; i++) {
		double w1 = gray_levels_cum[i] / tot;
		double w2 = 1 - w1;
		double mu_1, mu_2;
		if (gray_levels_cum[i] == 0) {
			mu_1 = 0.0;
		}
		else {
			mu_1 = weighted_gray_levels_cum[i] / gray_levels_cum[i];
		}

		if (tot - gray_levels_cum[i] == 0) {
			mu_2 = 0.0;
		}else{
			mu_2 = (weighted_tot - weighted_gray_levels_cum[i]) / 
				(tot - gray_levels_cum[i]);
		}

		double variance_B = w1 * pow(mu_1 - mean, 2.0) +
			w2 * pow(mu_2 - mean, 2.0);
		
		if (variance_B > mx) {
			mx = variance_B;
			I0 = i;
		}
	}
	
	d_threshold = I0;
	d_variance = variance_B;
}

int ParticleFilter::LoadImage(const string& file_name) {
	d_img = cv::imread(file_name, IMREAD_GRAYSCALE);
	if (d_img.empty()) {
		cout << "Could not read image file" << endl;
		return -1;
	}

	return 0;
}

void ParticleFilter::Denoise() {
	double stddev = 100.0;
	cv::GaussianBlur(d_img, d_denoised_img, Size(7, 7), stddev);

	const double zoom_scale = 4.0;
	cv::Mat resized_img;
	int small_rows = (int)zoom_scale * d_denoised_img.rows;
	int small_cols = (int)zoom_scale * d_denoised_img.cols;
	cv::resize(d_denoised_img, resized_img, Size(small_rows, small_cols;)
	
}

void ParticleFilter::FindGradients() {
	const int sobel_size = 11;
	cv::Sobel(d_denoised_img, d_grad_x, CV_32FC1, 1, 0, 11);
	cv::Sobel(d_denoised_img, d_grad_y, CV_32FC1, 0, 1, 11);

}