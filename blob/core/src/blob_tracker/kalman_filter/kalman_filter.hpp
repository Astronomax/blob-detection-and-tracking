#ifndef BLOB_TRACKER_KALMAN_FILTER_HPP
#define BLOB_TRACKER_KALMAN_FILTER_HPP

#include <boost/numeric/ublas/matrix.hpp>

class kalman_filter {
public:
	kalman_filter(float dt, float std_acc, float x_std_meas, float y_std_meas);
	std::pair<float, float> predict();
	std::pair<float, float> update(std::pair<float, float> p);
private:
	boost::numeric::ublas::matrix<float> x, A, H, Q, R, P;
};

#endif //BLOB_TRACKER_KALMAN_FILTER_HPP
