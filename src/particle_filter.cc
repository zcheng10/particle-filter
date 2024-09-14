#include "particle_filter.h"

using namespace std;

using pii = pair<int, int>;

void check_mat(const cv::Mat& m, const std::string& prompt = "") {
	cout << prompt << m.rows << ", " << m.cols << " " << m.type() << endl;
}

void ParticleFilter::Threshold() {
	int xi = 0;
	int xf = d_img.cols;
	int yi = 0;
	int yf = d_img.rows;

	cout << "image shape: ";
	cout << d_img.rows << " " << d_img.cols << " " << d_img.type() << " <-> ";
	cout << d_denoised_img.rows << " " << d_denoised_img.cols 
		<< " " << d_denoised_img.type() << endl;

	const int L = 256;
	vector<int> gray_levels(L, 0);
	for (int i = yi; i < yf; i++) {
		for (int j = xi; j < xf; j++) {
			int ind = (int)d_denoised_img.at<double>(i, j);
			ind = std::max(ind, 0);
			ind = std::min(ind, 255);

			/* if (ind < 0 || ind>255)
				cout << "wrong GLV " << ind << endl; */
			
			gray_levels[ind]++;
		}
	}

	vector<int> gray_levels_cum(L, 0);
	vector<double> weighted_gray_levels_cum(L, 0.0);
	double mean = 0.0;
	int tot = d_denoised_img.rows * d_denoised_img.cols;
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
		}
		else {
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

	d_I0 = I0;
	d_variance = mx;
}

bool ParticleFilter::AtEdge(int x, int y) {
	return ((x == 0) || (x == d_cols - 1)) && ((y == 0) || (y == d_rows - 1));
}

bool ParticleFilter::PixValid(int x, int y) {
	return (x >= 0) && (x < d_rows) && (y >= 0) && (y < d_cols);
}

int ParticleFilter::LoadImage(const string& file_name) {
	d_img = cv::imread(file_name, cv::IMREAD_GRAYSCALE); //  CV_32FC1);
	if (d_img.empty()) {
		cout << "Could not read image file" << endl;
		return -1;
	}

	Prepare();
	return 0;
}

void ParticleFilter::Denoise() {
	double stddev = 100.0;
	cv::Mat img8u;
	cv::GaussianBlur(d_img, img8u, cv::Size(7, 7), stddev);
	img8u.convertTo(d_denoised_img, CV_64F);

	const double zoom_scale = 4.0;
	cv::Mat resized_img;
	int small_rows = (int)zoom_scale * d_denoised_img.rows;
	int small_cols = (int)zoom_scale * d_denoised_img.cols;
	cv::resize(d_denoised_img, resized_img, cv::Size(small_cols, small_rows));
	cv::Mat big_img;
	cv::resize(resized_img, big_img, cv::Size(small_cols, small_rows));

}

void ParticleFilter::FindConstraints() {
	check_mat(d_grad_x, "grad_x: ");
	check_mat(d_grad_x, "grad_y: ");
	check_mat(d_constraints, "constraints: ");

	for (int i = 0; i < d_denoised_img.rows; i++) {
		for (int j = 0; j < d_denoised_img.cols; j++) {
			double pg = exp(-1.0 * pow(d_denoised_img.at<double>(i, j) -
				d_I0, 2.0) / d_variance);
			double grad = pow(d_grad_x.at<double>(i, j), 2.0) +
				pow(d_grad_y.at<double>(i, j), 2.0);
			double pl = grad / (grad + d_lambda);
			d_constraints.at<double>(i, j) = pg * pl;
		}
	}
}

bool ParticleFilter::FindStartpoint(pii& startpoint) {
	double mx = 0.0;
	int x = 0;
	int y = 0;
	startpoint = pii(-1, -1);
	for (int i = 0; i < d_denoised_img.cols; i++) {
		for (int j = 0; j < d_denoised_img.rows; j++) {
			if (d_constraints.at<double>(i, j) > mx) {
				x = i;
				y = j;
				mx = d_constraints.at<double>(i, j);
			}
		}
	}

	if (mx > d_I0) {
		startpoint = pii(x, y);
		return true;
	}

	return false;
}

bool ParticleFilter::FindNext() {
	bool done = true;
	vector<WeightTuple> weights;
	for (int i = 0; i < d_particles.size(); i++) {
		Particle& p = d_particles[i];
		if (p.complete1 && p.complete2) {
			weights.push_back(WeightTuple(p.weight, i, -1, -1, -1, -1));
			continue;
		}

		done = false;
		if (d_particles[i].complete1) {
			FindNextDir(i, 2, weights);
		}
		else {
			FindNextDir(i, 1, weights);
		}
	}

	if (done) {
		int fin = FindFinalParticle();
		d_contourpaths.push_back(d_particles[fin]);
		return true;
	}

	FindCandidates(weights);

	return done;
}

int ParticleFilter::FindVectorDirection(double x, double y) {
	double q2 = 1 / pow(2.0, 0.5);
	double step[8][2] = { {1., 0.}, {q2, q2}, {0., 1.}, {-q2, q2},
		{-1., 0.}, {-q2, -q2}, {0., -1.}, {q2, -q2} };

	double mx = x * step[0][0] + y * step[0][1];
	int ind = 0;
	for (int i = 1; i < 8; i++) {
		double dp = x * step[i][0] + y * step[i][1];
		if (dp > mx) {
			mx = dp;
			ind = i;
		}
	}

	return ind;
}

bool ParticleFilter::FindNextDir(int pi, int dir,
	vector<WeightTuple>& weights) {

	Particle& p = d_particles[pi];

	int step[8][2] = { {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0},
		{-1, -1}, {0, -1}, {1, -1} };

	bool& complete = (dir == 1) ? p.complete1 : p.complete2;
	pii& last = (dir == 1) ? p.last1 : p.last2;
	pii& penum = (dir == 1) ? p.penum1 : p.penum2;
	int& dirpenum = (dir == 1) ? p.dirpenum1 : p.dirpenum2;

	double grad_x = d_grad_x.at<double>(last.first, last.second);
	double grad_y = d_grad_y.at<double>(last.first, last.second);
	int index = FindVectorDirection(grad_x, grad_y);

	bool done = false;

	if (complete) {
		weights.push_back(WeightTuple(p.weight, pi, -1, -1, -1, -1));
		done = true;
		return done;
	}

	for (int i = 1; i < 3; i++) {
		int ind = (dir == 1) ? (index + i) % 8 : (index - i + 8) % 8;
		int xit1 = last.first + step[ind][0];
		int xit2 = last.second + step[ind][1];
		if (xit1 == penum.first && xit2 == penum.second) {
			index = (index + 4) % 8;
			break;
		}
	}

	for (int i = 1; i < 3; i++) {
		int ind = (dir == 1) ? (index + i) % 8 : (index - i + 8) % 8;
		pii xit = pii(last.first + step[ind][0], last.second + step[i][1]);
		if (p.pix.find(xit) != p.pix.end()) {
			weights.push_back(WeightTuple(p.weight, pi, -1, -1, -1, dir));
			continue;
		}

		if (AtEdge(xit.first, xit.second)) {
			weights.push_back(WeightTuple(p.weight, pi, -2, -2, -1, dir));
			continue;
		}

		double logcons = -100000.0;
		if (d_constraints.at<double>(xit.first, xit.second) > 0) {
			logcons = log(d_constraints.at<double>(xit.first, xit.second));
		}
		double newweight = p.weight + logcons;
		weights.push_back(WeightTuple(newweight, pi, xit.first, xit.second,
			index, dir));
	}

	return done;
}

void ParticleFilter::FindCandidates(vector<WeightTuple>& weights) {
	sort(weights.rbegin(), weights.rend());

	vector<int> dist_ind;
	for (int i = 0; i < weights.size(); i++) {
		bool dist = true;

		if (weights[i].next.first == -1 && weights[i].next.second == -1) {
			dist_ind.push_back(i);
			continue;
		}

		for (int j = 0; j < dist_ind.size(); j++) {
			if (weights[i].next == weights[dist_ind[i]].next) {
				dist = false;
				break;
			}
		}

		if (dist) {
			dist_ind.push_back(i);
		}
	}

	int sz = min((int)dist_ind.size(), d_N);
	vector<Particle> cand(sz);
	for (int i = 0; i < sz; i++) {
		int k = dist_ind[i];
		int ind = weights[k].index;
		cand[i] = d_particles[ind];

		if (weights[k].next.first == -1 && weights[k].next.second == -1) {
			cand[i].complete1 = true;
			cand[i].complete2 = true;
			continue;
		}

		if (weights[k].next.first == -2 && weights[k].next.second == -2) {
			cand[i].complete1 = true;
			cand[i].complete2 = (weights[i].dir == 1) ? false : true;
			continue;
		}

		bool& complete = (weights[k].dir == 1) ? cand[i].complete1 :
			cand[i].complete2;
		pii& last = (weights[k].dir == 1) ? cand[i].last1 : cand[i].last2;
		pii& penum = (weights[k].dir == 1) ? cand[i].penum1 :
			cand[i].penum2;
		int& dirpenum = (weights[k].dir == 1) ? cand[i].dirpenum1 :
			cand[i].dirpenum2;

		int x = last.first;
		int y = last.second;
		penum = pii(x, y);
		last = weights[k].next;
		dirpenum = weights[k].dirpenum;
		cand[i].pix.insert(pii(x, y));
	}

	d_particles.clear();
	d_particles.resize(cand.size());
	for (int i = 0; i < cand.size(); i++) {
		d_particles[i] = cand[i];
	}
}

int ParticleFilter::FindFinalParticle() {
	double mnweight = d_particles[0].weight;
	int ind = 0;
	for (int i = 1; i < d_particles.size(); i++) {
		if (d_particles[i].weight > mnweight) {
			mnweight = d_particles[i].weight;
			ind = i;
		}
	}

	return ind;
}

void ParticleFilter::FindGradients() {
	const int sobel_size = 11;
	cv::Sobel(d_denoised_img, d_grad_x, CV_64F, 1, 0, 11);
	cv::Sobel(d_denoised_img, d_grad_y, CV_64F, 0, 1, 11);
}

void ParticleFilter::Clear() {
	d_img.release();
	d_denoised_img.release();
	d_grad_x.release();
	d_grad_y.release();
	d_constraints.release();
}

void ParticleFilter::Prepare() {
	d_denoised_img = d_img;
	d_grad_x = d_img;
	d_grad_y = d_img;
	d_constraints = cv::Mat(d_img.rows, d_img.cols, CV_64F); //  d_img;
}

void ParticleFilter::InitializeParticles(pii& startpoint) {
	int x = startpoint.first;
	int y = startpoint.second;
	int ind = FindVectorDirection(d_grad_x.at<double>(x, y),
		d_grad_y.at<double>(x, y));
	int step[8][2] = { {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0},
		{-1, -1}, {0, -1}, {1, -1} };
	d_particles.resize(d_N);
	for (int i = 1; i <= d_N; i++) {
		pii next = { -1, -1 };
		next.first = x + step[ind + i][0];
		next.second = y + step[ind + i][1];
		double weight = log(d_constraints.at<double>(next.first, next.second))
			+ log(d_constraints.at<double>(x, y));
		if (!PixValid(next.first, next.second)) {
			next.first = x - step[ind + i][0];
			next.second = y - step[ind + i][1];
			d_particles[i - 1] = Particle(next, startpoint, 2, weight);
			continue;
		}

		d_particles[i - 1] = Particle(next, startpoint, 1, weight);
	}
}

void ParticleFilter::FindContours(const string filename) {
	Clear();
	LoadImage(filename);
	cout << "LoadImage done" << endl;

	Denoise();
	cout << "Denoise done" << endl;

	Threshold();
	cout << "Threshold done" << endl;

	FindGradients();
	cout << "FindGradients done" << endl;

	FindConstraints();
	cout << "FindConstraints done" << endl;

	bool done = false;
	while (!done) {
		pii startpoint = { -1, -1 };
		bool done = FindStartpoint(startpoint);
		if (!false) {
			break;
		}

		InitializeParticles(startpoint);

		bool donefinding = false;
		while (!donefinding) {
			FindNext();
		}
	}
}

vector<Particle> ParticleFilter::GetContourPaths() {
	return d_contourpaths;
}