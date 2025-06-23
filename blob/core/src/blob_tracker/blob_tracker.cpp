#include "blob_tracker.hpp"
#include <cassert>
#include <vector>
#include <ctime>
#include <numeric>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
template<class T>
using matrix = boost::numeric::ublas::matrix<T>;

namespace blobs {
	struct position {
		double i, j;
	};

	blob_tracker::blob_tracker(bool use_prediction) : m_use_prediction(use_prediction)
	{
		std::vector<unsigned int> ids(4 * MAXN);
		std::iota(ids.begin(), ids.end(), 0);
		std::copy(ids.begin(), ids.end(), std::inserter(id_pool, id_pool.begin()));
	}

	blob_tracker::internal_state::internal_state() :
		kf(0.1, 1, 0.1, 0.1), ttl(0) {}

	std::vector<blob_tracker::object>
	blob_tracker::track
		(const std::vector<blob<double>> &blobs,
		double move_threshold,
		double scale_threshold)
	{
		size_t n = state.size();
		size_t m = blobs.size();
		matrix<double> g(n, m);
		matrix<bool> c(n, m);

		std::vector<position> predictions(n);
		for (ptrdiff_t i = 0; i < n; i++) {
			if(m_use_prediction) {
				auto p = state[i].kf.predict();
				predictions[i].i = p.first;
				predictions[i].j = p.second;
			} else {
				predictions[i].i = state[i].obj.blob_data.y;
				predictions[i].j = state[i].obj.blob_data.x;
			}
		}

		if (m > MAXN)
			throw std::invalid_argument("track: invalid argument!");
		std::fill(c.data().begin(), c.data().end(), true);
		for (ptrdiff_t i = 0; i < n; i++) {
			for (ptrdiff_t j = 0; j < m; j++) {
				auto di = (double) (predictions[i].i - (double)blobs[j].y);
				auto dj = (double) (predictions[i].j - (double)blobs[j].x);
				double distance = /*sqrtf*/(di * di + dj * dj);
				g(i, j) = distance;
				if (distance > move_threshold)
					c(i, j) = false;
				double scale = state[i].obj.blob_data.r / blobs[j].r;
				if (scale < 1.f / scale_threshold || scale_threshold < scale)
					c(i, j) = false;
			}
		}

		auto matching = hungarian::hungarian(g, c);

		std::vector<object> objects;
		std::vector<internal_state> next_internal_state;
		std::vector<bool> born(m, true);

		for (ptrdiff_t i = 0; i < n; i++) {
			auto obj_data = state[i];
			if (matching[i] == -1) {
				if (obj_data.obj.status == o_status::ALIVE ||
					obj_data.obj.status == o_status::BORN)
				{
					//obj_data.ttl = 10;
					//obj_data.obj.status = o_status::GHOST;
					obj_data.obj.status = o_status::DIED;
					id_pool.insert(obj_data.obj.id);
				} else if (obj_data.obj.status == o_status::GHOST) {
					if (--obj_data.ttl == 0) {
						obj_data.obj.status = o_status::DIED;
						id_pool.insert(obj_data.obj.id);
					}
				}
				obj_data.obj.blob_data.y = (int)predictions[i].i;
				obj_data.obj.blob_data.x = (int)predictions[i].j;
			} else {
				born[matching[i]] = false;
				obj_data.obj.blob_data = blobs[matching[i]];
				obj_data.obj.status = o_status::ALIVE;
			}
			if (obj_data.obj.status != o_status::DIED) {
				next_internal_state.push_back(obj_data);
			}
			objects.push_back(obj_data.obj);
		}

		for (ptrdiff_t i = 0; i < m; i++) {
			if (born[i]) {
				internal_state obj_data;
				obj_data.obj = {o_status::BORN, *id_pool.begin(), blobs[i]};
				id_pool.erase(id_pool.begin());
				next_internal_state.push_back(obj_data);
				objects.push_back(obj_data.obj);
			}
		}
		state = next_internal_state;
		if(m_use_prediction) {
			for (auto &obj_data: state) {
				obj_data.kf.update({
					(double) obj_data.obj.blob_data.y,
					(double) obj_data.obj.blob_data.x
				});
			}
		}
		return objects;
	}
}