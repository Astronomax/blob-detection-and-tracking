#include "kalman_filter.hpp"

using matrix = boost::numeric::ublas::matrix<float>;

matrix
make_matrix(std::initializer_list<std::initializer_list<float>> list)
{
	size_t n = list.size(), m = list.begin()->size();
	matrix result(n, m);
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			result(i, j) = *((list.begin() + i)->begin() + j);
	return result;
}

kalman_filter::kalman_filter(float dt, float std_acc, float x_std_meas, float y_std_meas)
{
	x = make_matrix({{0}, {0}, {0}, {0}, {0}, {0}});
	A = make_matrix({
		{1.f, 0.f, dt, 0.f, powf(dt, 2.f) / 2.f, 0.f},
		{0.f, 1.f, 0.f, dt, 0.f, powf(dt, 2.f) / 2.f},
		{0.f, 0.f, 1.f, 0.f, dt, 0.f},
		{0.f, 0.f, 0.f, 1.f, 0.f, dt},
		{0.f, 0.f, 0.f, 0.f, 1.f, 0.f},
		{0.f, 0.f, 0.f, 0.f, 0.f, 1.f}
	});
	H = make_matrix({{1.f, 0.f, 0.f, 0.f, 0.f, 0.f},
					 {0.f, 1.f, 0.f, 0.f, 0.f, 0.f}});
	Q = make_matrix({
		{powf(dt, 4.f) / 4.f,	0.f,					powf(dt, 3.f) / 2.f,	0.f,					powf(dt, 2.f) / 2.f,	0.f},
		{0.f,					powf(dt, 4.f) / 4.f,	0.f,					powf(dt, 3.f) / 2.f,	0.f, 					powf(dt, 2.f) / 2.f},
		{powf(dt, 3.f) / 2.f,	0.f,					powf(dt, 2.f),			0.f,					dt,						0.f},
		{0.f,					powf(dt, 3.f) / 2.f,	0.f,					powf(dt, 2.f),			0.f,					dt},
		{powf(dt, 2.f) / 2.f,	0.f,					dt,						0.f,					1.f,					0.f},
		{0.f,					powf(dt, 2.f) / 2.f,	0.f,					dt,						0.f,					1.f}}) * powf(std_acc, 2.f);
	R = make_matrix({{powf(x_std_meas, 2.f), 0.f},
					 {0.f, powf(y_std_meas, 2.f)}});
	P = boost::numeric::ublas::identity_matrix(A.size2()) * 100.f;
}

std::pair<float, float>
kalman_filter::predict()
{
	x = prod(A, x);
	P = prod(matrix(prod(A, P)), trans(A)) + Q;
	return std::make_pair(x(0, 0), x(1, 0));
}

matrix
inverse(const matrix& m)
{
	matrix n(2, 2);
	const float a = m(0, 0);
	const float b = m(0, 1);
	const float c = m(1, 0);
	const float d = m(1, 1);
	const float determinant = 1.f / ((a * d) - (b * c));
	n(0, 0) =  d * determinant;
	n(0, 1) = -b * determinant;
	n(1, 0) = -c * determinant;
	n(1, 1) =  a * determinant;
	return n;
}

std::pair<float, float>
kalman_filter::update(std::pair<float, float> p)
{
	auto z = make_matrix({{p.first}, {p.second}});
	matrix S = prod(H, matrix(prod(P, trans(H)))) + R;
	auto inv_S = inverse(S);
	matrix K = prod(matrix(prod(P, trans(H))), inv_S);
	x = (x + prod(K, (z - matrix(prod(H, x)))));
	matrix I = boost::numeric::ublas::identity_matrix(H.size2());
	P = prod((I - prod(K, H)), P);
	return std::make_pair(x(0, 0), x(1, 0));
}
