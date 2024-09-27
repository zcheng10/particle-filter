#include "particle_filter.h"

using namespace std;

using pii = pair<int, int>;

void check_mat(const cv::Mat& m, const std::string& prompt = "") {
	cout << prompt << m.rows << ", " << m.cols << " " << m.type() << endl;
}

void ParticleFilter::Threshold() {
	cout << "image shape: ";
	cout << d_img.rows << " " << d_img.cols << " " << d_img.type() << " <-> ";
	cout << d_denoised_img.rows << " " << d_denoised_img.cols 
		<< " " << d_denoised_img.type() << endl;

	int otsuThreshValue;
	cv::Mat binaryImg;
	otsuThreshValue = cv::threshold(d_img, binaryImg, 0, 255, 
		cv::THRESH_BINARY | cv::THRESH_OTSU);

	cv::Scalar mean, stddev;
	d_I0 = otsuThreshValue;
	cv::meanStdDev(d_denoised_img, mean, stddev);
	d_variance = static_cast<double>(stddev[0]);
}

bool ParticleFilter::AtEdge(int x, int y) {
	return (x == d_area.x_start) || (x == d_area.x_end - 1) || (y == d_area.y_start)
		|| (y == d_area.y_end - 1);
}

bool ParticleFilter::PixValid(int x, int y) {
	return (x >= d_area.x_start) && (x < d_area.x_end) && (y >= d_area.y_start) &&
		(y < d_area.y_end);
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

	cv::Mat resized_img; cout << __LINE__ << endl;
	int small_rows = (int)d_zoom_scale * d_denoised_img.rows;
	int small_cols = (int)d_zoom_scale * d_denoised_img.cols;
	cv::resize(d_denoised_img, resized_img, cv::Size(small_cols, small_rows));

	cv::Mat big_img, big_img2;
	cv::resize(resized_img, big_img, cv::Size(d_cols, d_rows));
	big_img.convertTo(big_img2, CV_64F);
	d_denoised_img = d_zoom_weight * big_img2 + (1 - d_zoom_weight) * d_denoised_img;
}

void ParticleFilter::FindConstraints() {
	check_mat(d_grad_x, "grad_x: ");
	check_mat(d_grad_x, "grad_y: ");
	check_mat(d_constraints, "constraints: ");

	for (int i = d_area.x_start; i < d_area.x_end; i++) {
		for (int j = d_area.y_start; j < d_area.y_end; j++) {
			double pg = exp(-1.0 * pow(d_denoised_img.at<double>(i, j) -
				d_I0, 2.0) / d_variance);
			double grad = pow(d_grad_x.at<double>(i, j), 2.0) +
				pow(d_grad_y.at<double>(i, j), 2.0);
			double pl = grad / (grad + d_lambda);
			d_constraints.at<double>(i, j) = pg * pl;

			if (pg * pl == 0) {
				d_logconstraints.at<double>(i, j) = -100000.0;
				continue;
			}

			d_logconstraints.at<double>(i, j) = log(pg * pl);
		}
	}
}

bool ParticleFilter::FindStartpoint(pii& startpoint) {
	double mx = 0.0;
	int x = 0;
	int y = 0;
	startpoint = pii(-1, -1);
	for (int i = d_area.x_start; i < d_area.x_end; i++) {
		for (int j = d_area.y_start; j < d_area.y_end; j++) {
			if (d_found[i][j]) {
				continue;
			}

			if (d_constraints.at<double>(i, j) > mx) {
				x = i;
				y = j;
				mx = d_constraints.at<double>(i, j);
			}
		}
	}

	cout << "threshold = " << d_threshold << endl;
	cout << "mx = " << mx << endl;
	cout << "mx > d_threshold = " << (mx > d_threshold) << endl;

	if (mx > d_threshold) {
		startpoint.first = x;
		startpoint.second = y;
		return true;
	}

	return false;
}

bool ParticleFilter::FindNext() {
	bool done = true;
	vector<WeightTuple> weights;
	for (int i = 0; i < d_particles.size(); i++) {
		Particle& p = d_particles[i];
	//	p.print();
	//	cout << "---------------------------" << endl;
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

//	cout << "weights size = " << weights.size() << endl;
//	cout << "done: " << done << endl;
//	cout << "____________________________________________________" << endl;
	if (done) {
		int fin = FindFinalParticle();
		cout << "Final: " << endl;
		d_particles[fin].print();
		cout << "----------------------------------" << endl;
		d_contourpaths.push_back(d_particles[fin]);
		return done;
	}

	FindCandidates(weights);

	return done;
}


/** the direction is a value between 0 and 7, i.e. 8 directions
 */
int ParticleFilter::FindVectorDirection(double x, double y) {
	const double q2 = 1 / pow(2.0, 0.5);
	const double step[8][2] = { {1., 0.}, {q2, q2}, {0., 1.}, {-q2, q2},
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


bool ParticleFilter::FindNextDir(int pi, int dir, vector<WeightTuple>& weights) {
	Particle& p = d_particles[pi];
	const int step[8][2] = { {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0},
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

	for (int i = 1; i <= 3; i++) {
		int ind = (dir == 1) ? (index + i) % 8 : (index - i + 8) % 8;
		int xit1 = last.first + step[ind][0];
		int xit2 = last.second + step[ind][1];// cout << __LINE__ << endl;
		if (xit1 == penum.first && xit2 == penum.second) {
			index = (index + 4) % 8;
			break;
		}
	}

	//int d = dirpenum;
	//MakeGradientsContinuous(index, d);

	for (int i = 1; i <= 3; i++) {
		int ind = (dir == 1) ? (index + i) % 8 : (index - i + 8) % 8;// cout << __LINE__ << ": " << ind << endl;
		pii xit = pii(last.first + step[ind][0], last.second + step[i][1]);
	//	cout << __LINE__ << ": " << xit.first << " " << xit.second << endl;
		if (p.pix.find(xit) != p.pix.end()) {
			weights.push_back(WeightTuple(p.weight, pi, -1, -1, -1, dir)); 
		//	cout << __LINE__ << endl;
			continue;
		}

		if (AtEdge(xit.first, xit.second) || !PixValid(xit.first, xit.second)) {
			weights.push_back(WeightTuple(p.weight, pi, -2, -2, -2, dir));// cout << __LINE__ << endl;
			continue;
		}

		double newweight = p.weight + d_logconstraints.at<double>(xit.first, xit.second);
		weights.push_back(WeightTuple(newweight, pi, xit.first, xit.second,
			index, dir));// cout << __LINE__ << endl;
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
			if (weights[i].next == weights[dist_ind[j]].next) {
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
		//	cout << "weights[i].dir = " << weights[i].dir << endl;
		//	cout << "322: " << cand[i].complete2 << endl;
			continue;
		}

		bool& complete = (weights[k].dir == 1) ? cand[i].complete1 : cand[i].complete2;
		pii& last = (weights[k].dir == 1) ? cand[i].last1 : cand[i].last2;
		pii& penum = (weights[k].dir == 1) ? cand[i].penum1 :
			cand[i].penum2;
		int& dirpenum = (weights[k].dir == 1) ? cand[i].dirpenum1 :
			cand[i].dirpenum2;

		int x = last.first;
		int y = last.second;
		penum = pii(x, y);
		last.first = weights[k].next.first;
		last.second = weights[k].next.second;
		dirpenum = weights[k].dirpenum;
		cand[i].pix.insert(weights[k].next);
		cand[i].weight = weights[k].weight;
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

	for (auto& it : d_particles[ind].pix) {
		const int wid = 5;
		for (int j = -1 * wid; j <= wid; j++) {
			for (int k = -1 * wid; k <= wid; k++) {
				if (PixValid(it.first + j, it.second + k)) {
					d_found[it.first + j][it.second + k] = true;
				}
			}
		}
	}

	return ind;
}

void ParticleFilter::MakeGradientsContinuous(int& graddir, int& prevgraddir) {
	for (int i = -1; i <= 1; i++) {
		if (graddir == (prevgraddir + i + 8) % 8) {
			return;
		}
	}

	int c1 = (prevgraddir - 1 + 8) % 8;
	int c2 = (prevgraddir + 1) % 8;
	int dist1 = 0;
	int dist2 = 0;
	if (c1 > graddir) {
		dist1 = min(c1 - graddir, graddir + 8 - c1);
	}
	else {
		dist1 = min(graddir - c1, c1 + 8 - graddir);
	}

	if (c2 > graddir) {
		dist2 = min(c2 - graddir, graddir + 8 - c2);
	}
	else {
		dist2 = min(graddir - c2, c2 + 8 - graddir);
	}

	if (dist1 < dist2) {
		graddir = c1;
	}
	else {
		graddir = c2;
	}
}

void ParticleFilter::Convolve(cv::Mat& m, cv::Mat& res, vector<vector<double> >& kern) {
	const int SIZE = kern.size();
	for (int i = SIZE / 2; i < m.rows - SIZE / 2; i++) {
		for (int j = SIZE / 2; j < m.cols - SIZE / 2; j++) {
			res.at<double>(i, j) = 0.0;
			for (int k1 = -SIZE / 2; k1 <= SIZE / 2; k1++) {
				for (int k2 = -SIZE / 2; k2 <= SIZE / 2; k2++) {
					res.at<double>(i, j) += kern[k1 + SIZE / 2][k2 + SIZE / 2] * 
						m.at<double>(i + k1, j + k2);
				}
			}
		}
	}
}

void ParticleFilter::FindGradients() {
	const int sobel_size = 3;
	vector<vector<double> > ykern = { {-1., 0., 1.}, {-2., 0., 2.}, {-1., 0., 1.} };
	vector<vector<double> > xkern = { {-1., -2., -1.}, {0., 0., 0.}, {1., 2., 1.} };
	Convolve(d_denoised_img, d_grad_x, xkern);
	Convolve(d_denoised_img, d_grad_y, ykern);
	d_area.shrinkBy(sobel_size / 2);
}


void ParticleFilter::Clear() {
	d_img.release();
	d_denoised_img.release();
	d_grad_x.release();
	d_grad_y.release();
	d_constraints.release();
}


void ParticleFilter::Prepare() {
	d_cols = d_img.cols;
	d_rows = d_img.rows;
	d_area = Bbox(0, d_rows, 0, d_cols);
	d_denoised_img = cv::Mat(d_img.rows, d_img.cols, CV_64F);
	d_grad_x = cv::Mat::zeros(d_img.rows, d_img.cols, CV_64F);
	d_grad_y = cv::Mat::zeros(d_img.rows, d_img.cols, CV_64F);
	d_constraints = cv::Mat::zeros(d_img.rows, d_img.cols, CV_64F); //  d_img;
	d_logconstraints = cv::Mat::zeros(d_img.rows, d_img.cols, CV_64F);
	d_found.resize(d_rows, vector<bool>(d_cols, false));
}


void ParticleFilter::InitializeParticles(pii& startpoint) {
	int x = startpoint.first;
	int y = startpoint.second;
	int ind = FindVectorDirection(d_grad_x.at<double>(x, y),
		d_grad_y.at<double>(x, y));
	
	const int step[8][2] = { {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0},
		{-1, -1}, {0, -1}, {1, -1} };
	
	d_particles.resize(d_N);

	for (int i = 1; i <= d_N; i++) {
		pii next = { -1, -1 };
		next.first = x + step[(ind + i) % 8][0];
		next.second = y + step[(ind + i) % 8][1];
		
		if (!PixValid(next.first, next.second)) {
			next.first = x - step[(ind + i) % 8][0];
			next.second = y - step[(ind + i) % 8][1];
			double weight = d_logconstraints.at<double>(next.first, next.second)
				+ d_logconstraints.at<double>(x, y);
			d_particles[i - 1] = Particle(next, startpoint, 2, weight);
			continue;
		}

		double weight = d_logconstraints.at<double>(next.first, next.second)
			+ d_logconstraints.at<double>(x, y);

		d_particles[i - 1] = Particle(next, startpoint, 1, weight);
	}
	/*
	for (int i = 0; i < d_N; i++) {
		d_particles[i].print();
	}*/
}


void ParticleFilter::FindContours(const string filename) {
	Clear();
	LoadImage(filename);
	cout << "LoadImage done" << endl;
	cout << "d_rows: " << d_rows << endl;
	cout << "d_cols: " << d_cols << endl;

	Denoise();
	cout << "Denoise done" << endl;

	Threshold();
	cout << "Threshold done" << endl;

	FindGradients();
	cout << "FindGradients done" << endl;

	FindConstraints();
	cout << "FindConstraints done" << endl;

	cv::Mat constraintoutput = cv::Mat::zeros(d_rows, d_cols, CV_64F);
	cout << "I0 = " << d_I0 << endl;
	cout << "mean denoised = " << cv::mean(d_denoised_img) << endl;
	
	double mxcn = 0.0;
	for (int i = d_area.x_start; i < d_area.x_end; i++) {
		for (int j = d_area.y_start; j < d_area.y_end; j++) {
			constraintoutput.at<double>(i, j) = 256.0 * d_constraints.at<double>(i, j);
			mxcn = min(d_logconstraints.at<double>(i, j), mxcn);
		}
	}

	cout << "min log constraint = " << mxcn << endl;

	cv::imwrite("test/Thresholded.jpg", constraintoutput);

	bool done = false;
	while (!done) {
		pii startpoint = { -1, -1 };
		bool found = FindStartpoint(startpoint);

		cout << "found = " << found << endl;
		if (!found) {
			break;
		}

		cout << "Startpoint: " << startpoint.first << ", " << startpoint.second;
		cout << "constraint: " << d_constraints.at<double>(startpoint.first, startpoint.second);
		cout << endl;

		InitializeParticles(startpoint);
		cout << " -> initialized particles" << endl;

		bool donefinding = false;
		while (!donefinding) {
			donefinding = FindNext();
		}

		cout << "contour found" << endl;
	}

	cout << "Checking for invalid pixels: " << endl;
	for (auto it : d_contourpaths[0].pix) {
		if (!PixValid(it.first, it.second)) {
			cout << "Not valid pixel: " << it.first << ", " << it.second << endl;
		}
	}
}

vector<Particle> ParticleFilter::GetContourPaths() {
	return d_contourpaths;
}