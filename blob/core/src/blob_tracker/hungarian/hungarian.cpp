#include "hungarian.hpp"
#include <stdexcept>

using namespace boost::numeric::ublas;

namespace hungarian {
	const float MAXW = 1e5;
	const float INF1 = 1e9;
	const float INF2 = 1e18;

	std::vector<ptrdiff_t>
	hungarian(const matrix<float> &g)
	{
		size_t n = g.size1();
		if (n == 0) return {};
		size_t m = g.size2();

		matrix<float> a(n + 1, m + 1);
		for (int i = 1; i <= n; i++)
			for (int j = 1; j <= m; j++)
				a(i, j) = g(i - 1, j - 1);

		bool transposed = false;
		if (n > m) {
			a = trans(a);
			std::swap(n, m);
			transposed = true;
		}
		/* http://e-maxx.ru/algo/assignment_hungary */
		std::vector<float> u(n + 1), v(m + 1);
		std::vector<ptrdiff_t> p(m + 1), way(m + 1);
		for (ptrdiff_t i = 1; i <= n; ++i) {
			p[0] = i;
			ptrdiff_t j0 = 0;
			std::vector<float> minv(m + 1, INF2);
			std::vector<bool> used(m + 1, false);
			do {
				used[j0] = true;
				ptrdiff_t i0 = p[j0], j1;
				float delta = INF2;
				for (ptrdiff_t j = 1; j <= m; ++j)
					if (!used[j]) {
						float cur = a(i0, j) - u[i0] - v[j];
						if (cur < minv[j])
							minv[j] = cur, way[j] = j0;
						if (minv[j] < delta)
							delta = minv[j], j1 = j;
					}
				for (ptrdiff_t j = 0; j <= m; ++j)
					if (used[j])
						u[p[j]] += delta, v[j] -= delta;
					else
						minv[j] -= delta;
				j0 = j1;
			} while (p[j0] != 0);
			do {
				int j1 = way[j0];
				p[j0] = p[j1];
				j0 = j1;
			} while (j0);
		}


		if(transposed) {
			std::vector<ptrdiff_t> matching(m);
			for(ptrdiff_t j = 1; j <= m; j++)
				matching[j - 1] = p[j] - 1;
			return matching;
		}
		else {
			std::vector<ptrdiff_t> matching(n);
			for (ptrdiff_t j = 1; j <= m; ++j)
				if (p[j] != 0)
					matching[p[j] - 1] = j - 1;
			return matching;
		}
	}

	std::vector<ptrdiff_t>
	hungarian(matrix<float> g, const matrix<bool> &c)
	{
		size_t n = g.size1();
		if (n == 0) return {};
		size_t m = g.size2();

		for (ptrdiff_t i = 0; i < n; i++)
			for (ptrdiff_t j = 0; j < m; j++)
				if (!c(i, j)) g(i, j) = INF1;
		auto matching = hungarian(g);
		for (ptrdiff_t i = 0; i < matching.size(); i++)
			if(matching[i] != -1 && !c(i, matching[i]))
				matching[i] = -1;
		return matching;
	}
}
