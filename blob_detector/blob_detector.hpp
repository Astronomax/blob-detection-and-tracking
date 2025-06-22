#ifndef BLOB_TRACKER_BLOB_DETECTOR_HPP
#define BLOB_TRACKER_BLOB_DETECTOR_HPP

#include <vector>
#include <cstdint>
#include "../blob/blob.hpp"
#include "kdtree/kdtree.hpp"
#include "fft/fft_fast.hpp"
#include "fft/fft_slow.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/pool/pool_alloc.hpp>

namespace blobs {
	using namespace boost::numeric::ublas;

	struct matrix_shape {
		size_t rows, cols;
	};

	/*! Метод свертки */
	enum class conv_method {
		AUTO, 		/*!<Максимально быстрый метод, использует SIMPLE для небольших фильтров,
				 	 *	и FFT_FAST для больших фильтров, засчет чего достигается высокая скорость. */
		SIMPLE,		/*!<Реализация свертки "в лоб", для небольших фильров работает достаточно быстро. */
		FFT_SLOW,	/*!<Реализация свертки через дискретное проеобразование Фурье.
				 	 *	Более медленная реализация преобразования Фурье, чем FFT_FAST, однако
				 	 *	в отличие от FFT_FAST обладает достаточно хорошей точностью. */
		FFT_FAST	/*!<Реализация свертки через дискретное проеобразование Фурье.
				 	 *	Работает значительно быстрее, чем FFT_FAST, однако имеет серьезный недостаток:
				 	 *	точность сильно зависит от размеров сворачиваемого изображения
				 	 *	(чем больше сворачиваемое изображение, тем сильнее страдает точность).*/
	};

	/**
	 * \brief Singleton-класс, реализующий LoG-фильтр. Обнаруживает контрастные участки на изображении,
	 * определяя их местоположение и приблизительный "радиус".
	 * 
	 * \tparam FPT 		floating point type (тип, используемый внутри класса в качестве fpt).
	 * \tparam CONV_METHOD 	Метод, который будет использоваться при свёртке. (см. conv_method)
	 * \tparam USE_SCALING	Булевый флаг, который отвечает за то, как будет строиться scale space.
	 *			При использовании масштабирования, ядро свертки всегда одно,
	 *			меняется только размер картинки пропорционально текущей sigma.
	 *			Если масштабирование не используется, scale space строится "честно" -
	 *			изображение в исходном масштабе сворачивается с ядром, соответствующим текущей sigma.
	 *			Использование масштабирования ведет к значительному улучшению
	 *			производительности, но при этом, конечно, страдает точность.
	 */
	template<typename FPT = float, conv_method CONV_METHOD = conv_method::AUTO, bool USE_SCALING = true>
	class blob_detector {
	public:
		blob_detector() = delete;

		/**
		 * \brief Метод обнаруживает пятна на изображении, используя Лаплассиан Гауссиана.
		 *  Для каждого найденного пятна на изображении метод возвращает координаты и примерный радиус пятна.
		 * 
		 * \param image		Входное монохромное изображение.
		 * 			Предполагается, что пятна - это светлые области на темном фоне.
		 * \param min_sigma	Минимальное стандартное отклонение ядра Гауссиана.
		 * 			Чем меньше эта величина, тем более мелкие пятна могут обнаруживаться.
		 * \param max_sigma	Максимальное стандартное отклонение ядра Гауссиана.
		 * 			Чем больше эта величина, тем более крупные пятна могут обнаруживаться.
		 * \param num_sigma	Количество промежуточных значений стандартных отклонений
		 * 			между `min_sigma` и `max_sigma`.
		 * \param threshold_abs	Нижняя граница максимумов в scale_space.
		 * 			Если значение в точке локальные максимума меньше чем `threshold`, то
		 * 			данная точка игнорируются. Чем меньше эта величина, тем более низкий
		 * 			порог интенсивности используется, тем больше пятен будет обнаружено.
		 * \param overlap	Значение от 0 до 1.
		 * 			Если площадь пересечения двух объектов занимает от площади наименьшего
		 * 			объекта долю, большую чем overlap, то меньший объект удаляется.
		 * \return	Массив обнаруженных объеков.
		 */
		static std::vector<blob<FPT>>
		detect_blobs(const matrix<FPT> &image, FPT min_sigma, FPT max_sigma, size_t num_sigma,
			FPT threshold_abs, FPT overlap, bool use_prune_blobs)
		{
			matrix_shape in_shape = {image.size1(), image.size2()};

			std::vector<FPT> sigma(num_sigma);
			std::vector<FPT> radius(num_sigma);

			for (ptrdiff_t k = 0; k < num_sigma; k++) {
				if(k == 0) {
					sigma[k] = min_sigma;
				} else {
					FPT coeff = (max_sigma - min_sigma) / (FPT) (num_sigma - 1);
					sigma[k] = min_sigma + (FPT) k * coeff;
				}
				radius[k] = sqrtf(2.f) * sigma[k];
			}

			auto scale_space = get_scale_space(image, sigma);
			std::vector<matrix<FPT>> max_in_scale_space(num_sigma);
			#pragma omp parallel for default(none) shared(num_sigma, in_shape, scale_space, max_in_scale_space)
			for (ptrdiff_t k = 0; k < num_sigma; k++) {
				max_in_scale_space[k] = matrix<FPT>(in_shape.rows, in_shape.cols);
				maximum_filter2d(max_in_scale_space[k], scale_space[k], 3);
			}

			std::vector<blob<FPT>> blobs;
			for (ptrdiff_t i = 0; i < in_shape.rows; i++) {
				for (ptrdiff_t j = 0; j < in_shape.cols; j++) {
					for (ptrdiff_t k = 0; k < num_sigma; k++) {
						FPT prev_max = ((k > 0) ? max_in_scale_space[k - 1](i, j) : -INF);
						FPT cur_max = max_in_scale_space[k](i, j);
						FPT next_max = ((k < num_sigma - 1) ? max_in_scale_space[k + 1](i, j) : -INF);
						if (cur_max > threshold_abs &&
							std::abs(scale_space[k](i, j) - cur_max) < EPS &&
							cur_max + EPS > prev_max && cur_max + EPS > next_max) {
							FPT r = radius[k];
							blobs.push_back({(int) i, (int) j, r});
						}
					}
				}
			}
			if(use_prune_blobs)
				return prune_blobs(blobs, overlap);
			return blobs;
		}

	protected:
		static void
		add_edge_padding(matrix<FPT> &dst, const matrix<FPT> &src,
			size_t top, size_t right, size_t bottom, size_t left) 
		{
			matrix_shape in_shape = { src.size1(), src.size2() };
			matrix_shape out_shape = {
				top + bottom + in_shape.rows,
				left + right + in_shape.cols
			};
			if (out_shape.rows != dst.size1() || out_shape.cols != dst.size2())
				throw std::invalid_argument("Add edge padding: invalid arguments!");
			for (ptrdiff_t i = 0; i < out_shape.rows; i++) {
				ptrdiff_t r = std::min(std::max((ptrdiff_t) top, i),
					(ptrdiff_t) (in_shape.rows + top - 1)) - (ptrdiff_t)top;
				for (ptrdiff_t j = 0; j < out_shape.cols; j++) {
					ptrdiff_t c = std::min(std::max((ptrdiff_t) left, j),
						(ptrdiff_t) (in_shape.cols + left - 1)) - (ptrdiff_t)left;
					dst(i, j) = src(r, c);
				}
			}
		}

		static void
		crop(matrix<FPT> &dst, const matrix<FPT> &src,
			size_t top, size_t right, size_t bottom, size_t left)
		{
			if (top + bottom + dst.size1() != src.size1() ||
				left + right + dst.size2() != src.size2())
			{
				throw std::invalid_argument("Crop: invalid arguments!");
			}
			size_t size1 = src.size1() - top - bottom;
			size_t size2 = src.size2() - left - right;
			for (ptrdiff_t i = 0; i < size1; i++)
				for (ptrdiff_t j = 0; j < size2; j++)
					dst(i, j) = src(top + i, left + j);
		}

		static void
		resize_image(matrix<FPT> &dst, const matrix<FPT> &src)
		{
			matrix_shape in_shape = {src.size1(), src.size2()};
			matrix_shape out_shape = {dst.size1(), dst.size2()};
			FPT S_R = (FPT)in_shape.rows / (FPT)out_shape.rows;
			FPT S_C = (FPT)in_shape.cols / (FPT)out_shape.cols;
			for (ptrdiff_t i = 0; i < out_shape.rows; i++) {
				for (ptrdiff_t j = 0; j < out_shape.cols; j++) {
					FPT rf = (FPT)i * S_R;
					FPT cf = (FPT)j * S_C;
					ptrdiff_t r = std::min((ptrdiff_t)in_shape.rows - 2, (ptrdiff_t)rf);
					ptrdiff_t c = std::min((ptrdiff_t)in_shape.cols - 2, (ptrdiff_t)cf);
					FPT delta_R = rf - (FPT)r;
					FPT delta_C = cf - (FPT)c;
					dst(i, j) = src(r, c) * (1.f - delta_R) * (1.f - delta_C) +
						src(r, c + 1) * (1.f - delta_R) * (delta_C) +
						src(r + 1, c) * (delta_R) * (1.f - delta_C) +
						src(r + 1, c + 1) * (delta_R) * (delta_C);
				}
			}
		}

		template<typename ITER1, typename ITER2>
		static void
		maximum_filter1d(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, size_t filter_size)
		{
			size_t n = end_src - beg_src;
			if(filter_size <= 3) {
				ptrdiff_t size1 = (ptrdiff_t)filter_size / 2;
				ptrdiff_t size2 = (ptrdiff_t)filter_size - size1;
				for (ptrdiff_t i = 0; i < n; i++) {
					ptrdiff_t l = std::max((ptrdiff_t)0, i - size1);
					ptrdiff_t r = std::min((ptrdiff_t)n, i + size2);
					*(beg_dst + i) = *(beg_src + l);
					for(ptrdiff_t j = l + 1; j < r; j++)
						*(beg_dst + i) = std::max(*(beg_dst + i), *(beg_src + j));
				}
			} else {
				ptrdiff_t lsp = 0, rsp = 0, wp = 0;
				std::vector<std::pair<FPT, ptrdiff_t>> stack(n);
				for (ptrdiff_t i = 0; i < n; i++) {
					ptrdiff_t rwp = i + ((ptrdiff_t) filter_size - 1) / 2;
					while (wp < n && wp <= rwp) {
						auto next_element = std::make_pair(*(beg_src + wp), wp);
						while (rsp > lsp && stack[rsp - 1].first < next_element.first)
							--rsp;
						stack[rsp] = std::make_pair(*(beg_src + wp), wp);
						++rsp;
						++wp;
					}
					if (stack[lsp].second < i - (ptrdiff_t) filter_size / 2)
						++lsp;
					*(beg_dst + i) = stack[lsp].first;
				}
			}
		}

		static void
		maximum_filter2d(matrix<FPT> &dst, const matrix<FPT> &src, size_t w_size)
		{
			for (ptrdiff_t i = 0; i < src.size1(); i++) {
				maximum_filter1d((dst.begin1() + i).begin(),
					(src.begin1() + i).begin(),
					(src.begin1() + i).end(),
					w_size
				);
			}
			for (ptrdiff_t j = 0; j < src.size2(); j++) {
				maximum_filter1d((dst.begin2() + j).begin(),
					(dst.begin2() + j).begin(),
					(dst.begin2() + j).end(),
					w_size
				);
			}
		}

		template<typename ITER1, typename ITER2>
		static void
		correlate1d_nearest_SIMPLE(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, const std::vector<FPT> &fw)
		{
			int symmetric = 0;
			ptrdiff_t filter_size = fw.size();
			ptrdiff_t size1 = filter_size / 2;
			ptrdiff_t size2 = filter_size - size1 - 1;
			ptrdiff_t n = end_src - beg_src;

			auto get_nearest = [](int idx, int len) {
				return std::min(std::max(0, idx), len - 1);
			};
			auto get_dst = [&](int idx) {
				return beg_dst + idx;
			};
			auto get_src_nearest = [&](int idx) {
				return *(beg_src + get_nearest(idx, n));
			};

			if (filter_size & 0x1) {
				symmetric = 1;
				for (ptrdiff_t i = 1; i <= filter_size / 2; i++) {
					if (fabs(fw[i + size1] - fw[size1 - i]) > EPS) {
						symmetric = 0;
						break;
					}
				}
				if (symmetric == 0) {
					symmetric = -1;
					for (ptrdiff_t i = 1; i <= filter_size / 2; i++) {
						if (fabs(fw[size1 + i] + fw[size1 - i]) > EPS) {
							symmetric = 0;
							break;
						}
					}
				}
			}
			if (symmetric > 0) {
				for (ptrdiff_t i = 0; i < n; i++) {
					*get_dst(i) = get_src_nearest(i) * fw[size1];
					for (ptrdiff_t j = -size1; j < 0; j++)
						*get_dst(i) += (get_src_nearest(i + j) + get_src_nearest(i - j)) * fw[j + size1];
				}
			} else if (symmetric < 0) {
				for (ptrdiff_t i = 0; i < n; i++) {
					*get_dst(i) = get_src_nearest(i) * fw[size1];
					for (ptrdiff_t j = -size1; j < 0; j++)
						*get_dst(i) += (get_src_nearest(i + j) - get_src_nearest(i - j)) * fw[j + size1];
				}
			} else {
				for (ptrdiff_t i = 0; i < n; i++) {
					*get_dst(i) = get_src_nearest(i + size2) * fw[size2 + size1];
					for (ptrdiff_t j = -size1; j < size2; j++)
						*get_dst(i) += get_src_nearest(i + j) * fw[j + size1];
				}
			}
		}

		template<typename ITER1, typename ITER2>
		static void
		correlate1d_nearest_FFT_SLOW(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, const std::vector<FPT> &fw)
		{
			size_t n = end_src - beg_src;
			std::vector<FPT> a(n + 2 * (fw.size() / 2));
			std::copy(beg_src, end_src, a.begin() + fw.size() / 2);
			for (int i = 0; i < fw.size() / 2; i++)
				a[i] = *beg_src;
			for (int i = 0; i < fw.size() / 2; i++)
				a[i + fw.size() / 2 + n] = *(end_src - 1);
			auto c = fft_slow<FPT>::multiply_polynomials(a, fw);
			std::copy(c.begin() + fw.size() - 1, c.begin() + fw.size() - 1 + n, beg_dst);
		}

		template<typename ITER1, typename ITER2>
		static void
		correlate1d_nearest_FFT_FAST(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, const std::vector<FPT> &fw)
		{
			size_t n = end_src - beg_src;
			std::vector<FPT> a(n + 2 * (fw.size() / 2));
			std::copy(beg_src, end_src, a.begin() + fw.size() / 2);
			for (int i = 0; i < fw.size() / 2; i++) a[i] = *beg_src;
			for (int i = 0; i < fw.size() / 2; i++) a[i + fw.size() / 2 + n] = *(end_src - 1);
			auto c = fft_fast<FPT>::multiply_polynomials(a, fw);
			std::copy(c.begin() + fw.size() - 1, c.begin() + fw.size() - 1 + n, beg_dst);
		}

		template<typename ITER1, typename ITER2>
		static void
		correlate1d_nearest(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, const std::vector<FPT> &fw)
		{
			switch (CONV_METHOD) {
			case conv_method::AUTO: {
				if(fw.size() < 40)
					correlate1d_nearest_SIMPLE(beg_dst, beg_src, end_src, fw);
				else 
					correlate1d_nearest_FFT_FAST(beg_dst, beg_src, end_src, fw);
				break;
			}
			case conv_method::SIMPLE: {
				correlate1d_nearest_SIMPLE(beg_dst, beg_src, end_src, fw);
				break;
			}
			case conv_method::FFT_SLOW: {
				correlate1d_nearest_FFT_SLOW(beg_dst, beg_src, end_src, fw);
				break;
			}
			case conv_method::FFT_FAST: {
				correlate1d_nearest_FFT_FAST(beg_dst, beg_src, end_src, fw);
				break;
			}
			}
		}

		template<typename ITER1, typename ITER2>
		static void
		correlate1d_SIMPLE(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, const std::vector<FPT> &fw)
		{
			int symmetric = 0;
			ptrdiff_t filter_size = fw.size();
			ptrdiff_t size1 = filter_size / 2;
			ptrdiff_t size2 = filter_size - size1 - 1;
			ptrdiff_t n = end_src - beg_src;

			auto get_dst = [&](int idx) {
				return beg_dst + idx;
			};
			auto get_src = [&](int idx) {
				return *(beg_src + idx);
			};

			if (filter_size & 0x1) {
				symmetric = 1;
				for (ptrdiff_t i = 1; i <= filter_size / 2; i++) {
					if (fabs(fw[i + size1] - fw[size1 - i]) > EPS) {
						symmetric = 0;
						break;
					}
				}
				if (symmetric == 0) {
					symmetric = -1;
					for (ptrdiff_t i = 1; i <= filter_size / 2; i++) {
						if (fabs(fw[size1 + i] + fw[size1 - i]) > EPS) {
							symmetric = 0;
							break;
						}
					}
				}
			}
			if (symmetric > 0) {
				for (ptrdiff_t i = size1; i < n - size1; i++) {
					*get_dst(i - size1) = get_src(i) * fw[size1];
					for (ptrdiff_t j = -size1; j < 0; j++)
						*get_dst(i - size1) += (get_src(i + j) + get_src(i - j)) * fw[j + size1];
				}
			} else if (symmetric < 0) {
				for (ptrdiff_t i = size1; i < n - size1; i++) {
					*get_dst(i - size1) = get_src(i) * fw[size1];
					for (ptrdiff_t j = -size1; j < 0; j++)
						*get_dst(i - size1) += (get_src(i + j) - get_src(i - j)) * fw[j + size1];
				}
			} else {
				for (ptrdiff_t i = size1; i < n - size2; i++) {
					*get_dst(i - size1) = 0;
					for (ptrdiff_t j = -size1; j <= size2; j++)
						*get_dst(i - size1) += get_src(i + j) * fw[j + size1];
				}
			}
		}

		template<typename ITER1, typename ITER2>
		static void
		correlate1d_FFT_SLOW(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, const std::vector<FPT> &fw)
		{
			size_t n = end_src - beg_src;
			std::vector<FPT> a(n);
			std::copy(beg_src, end_src, a.begin());
			auto c = fft_slow<FPT>::multiply_polynomials(a, fw);
			std::copy(c.begin() + fw.size() - 1, c.begin() + n, beg_dst);
		}

		template<typename ITER1, typename ITER2>
		static void
		correlate1d_FFT_FAST(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, const std::vector<FPT> &fw)
		{
			size_t n = end_src - beg_src;
			std::vector<FPT> a(n);
			std::copy(beg_src, end_src, a.begin());
			auto c = fft_fast<FPT>::multiply_polynomials(a, fw);
			std::copy(c.begin() + fw.size() - 1, c.begin() + n, beg_dst);
		}

		template<typename ITER1, typename ITER2>
		static void
		correlate1d(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, const std::vector<FPT> &fw)
		{
			switch (CONV_METHOD) {
			case conv_method::AUTO: {
				if(fw.size() < 40)
					correlate1d_SIMPLE(beg_dst, beg_src, end_src, fw);
				else correlate1d_FFT_FAST(beg_dst, beg_src, end_src, fw);
				break;
			}
			case conv_method::SIMPLE: {
				correlate1d_SIMPLE(beg_dst, beg_src, end_src, fw);
				break;
			}
			case conv_method::FFT_SLOW: {
				correlate1d_FFT_SLOW(beg_dst, beg_src, end_src, fw);
				break;
			}
			case conv_method::FFT_FAST: {
				correlate1d_FFT_FAST(beg_dst, beg_src, end_src, fw);
				break;
			}
			}
		}

		static std::vector<FPT>
		gaussian_kernel1d(FPT sigma, int order, int lw)
		{
			std::vector<FPT> exponent_range(order + 1);
			for (int i = 0; i <= order; i++)
				exponent_range[i] = i;
			FPT sigma2 = pow(sigma, 2.0);
			std::vector<FPT> phi_x(2 * lw + 1);
			for (int i = -lw; i <= lw; i++)
				phi_x[i + lw] = exp(-0.5 / sigma2 * pow(i, 2.0));
			FPT phi_sum = 0.0;
			for (int i = 0; i <= 2 * lw; i++)
				phi_sum += phi_x[i];
			for (int i = 0; i <= 2 * lw; i++)
				phi_x[i] /= phi_sum;
			if (order == 0) {
				return phi_x;
			} else {
				matrix<FPT> q(order + 1, 1);
				std::fill(q.data().begin(), q.data().end(), 0.f);
				q(0, 0) = 1;
				matrix<FPT> D(order + 1, order + 1);
				std::fill(D.data().begin(), D.data().end(), 0.f);
				for (int i = 0; i < order; i++)
					D(i, i + 1) = exponent_range[i + 1];
				matrix<FPT> P(order + 1, order + 1);
				std::fill(P.data().begin(), P.data().end(), 0.f);
				for (int i = 0; i < order; i++)
					P(i + 1, i) = 1.f / -sigma2;
				auto Q_deriv = D + P;
				for (int i = 0; i < order; i++)
					q = prod(Q_deriv, q);
				matrix<FPT> S(2 * lw + 1, order + 1);
				for (int i = 0; i <= 2 * lw; i++)
					for (int j = 0; j <= order; j++)
						S(i, j) = pow(i - lw, exponent_range[j]);
				q = prod(S, q);
				std::vector<FPT> res(2 * lw + 1);
				for (int i = 0; i <= 2 * lw; i++)
					res[i] = q(i, 0) * phi_x[i];
				return res;
			}
		}

		template<typename ITER1, typename ITER2>
		static void
		gaussian_filter1d(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, FPT sigma, int order)
		{
			int lw = 4.0 * sigma + 0.5;
			std::vector<FPT> weights = gaussian_kernel1d(sigma, order, (int)lw);
			correlate1d_nearest(beg_dst, beg_src, end_src, weights);
		}

		static void
		gaussian_laplace(matrix<FPT> &dst, const matrix<FPT> &src, FPT sigma)
		{
			std::fill(dst.data().begin(), dst.data().end(), 0.f);
			matrix<FPT> tmp(src.size1(), src.size2());
			matrix<FPT> tmp1(src.size1(), src.size2());
			#pragma omp parallel for default(none) shared(src, tmp, sigma)
			for (ptrdiff_t i = 0; i < src.size1(); i++) {
				gaussian_filter1d((tmp.begin1() + i).begin(),
					(src.begin1() + i).begin(),
					(src.begin1() + i).end(),
					sigma,
					0
				);
			}
			#pragma omp parallel for default(none) shared(src, tmp1, tmp, sigma)
			for (ptrdiff_t j = 0; j < src.size2(); j++) {
				gaussian_filter1d((tmp1.begin2() + j).begin(),
					(tmp.begin2() + j).begin(),
					(tmp.begin2() + j).end(),
					sigma,
					2
				);
			}
			dst += tmp1;
			#pragma omp parallel for default(none) shared(src, tmp, sigma)
			for (ptrdiff_t i = 0; i < src.size1(); i++) {
				gaussian_filter1d((tmp.begin1() + i).begin(),
					(src.begin1() + i).begin(),
					(src.begin1() + i).end(),
					sigma,
					2
				);
			}
			#pragma omp parallel for default(none) shared(src, tmp1, tmp, sigma)
			for (ptrdiff_t j = 0; j < src.size2(); j++) {
				gaussian_filter1d((tmp1.begin2() + j).begin(),
					(tmp.begin2() + j).begin(),
					(tmp.begin2() + j).end(),
					sigma,
					0
				);
			}
			dst += tmp1;
		}

		static void
		gaussian_laplace_alt_impl(matrix<FPT> &dst, matrix<FPT> &src, FPT sigma)
		{
			std::fill(dst.data().begin(), dst.data().end(), 0.f);

			int radius = 4.0 * sigma + 0.5;
			auto weights0 = gaussian_kernel1d(sigma, 0, (int)radius);
			auto weights2 = gaussian_kernel1d(sigma, 2, (int)radius);

			#pragma omp sections
			{
				#pragma omp section
				{
					matrix_shape row_padded_shape = {src.size1(), src.size2() + 2 * radius};
					matrix<FPT> row_padded(row_padded_shape.rows, row_padded_shape.cols);
					add_edge_padding(row_padded, src, 0, radius, 0, radius);
					matrix<FPT> row_tmp(row_padded_shape.rows, row_padded_shape.cols);
					matrix<FPT> row_tmp2(src.size1(), src.size2());
					correlate1d(row_tmp.data().begin() + radius, row_padded.data().begin(), row_padded.data().end(), weights0);
					crop(row_tmp2, row_tmp, 0, radius, 0, radius);

					matrix_shape col_padded_shape = {src.size2(), src.size1() + 2 * radius};
					matrix<FPT> col_padded(col_padded_shape.rows, col_padded_shape.cols);
					add_edge_padding(col_padded, trans(row_tmp2), 0, radius, 0, radius);
					matrix<FPT> col_tmp(col_padded_shape.rows, col_padded_shape.cols);
					matrix<FPT> col_tmp2(src.size2(), src.size1());
					correlate1d(col_tmp.data().begin() + radius, col_padded.data().begin(), col_padded.data().end(), weights2);
					crop(col_tmp2, col_tmp, 0, radius, 0, radius);

					dst += trans(col_tmp2);
				}
				#pragma omp section
				{
					matrix_shape row_padded_shape = {src.size1(), src.size2() + 2 * radius};
					matrix<FPT> row_padded(row_padded_shape.rows, row_padded_shape.cols);
					add_edge_padding(row_padded, src, 0, radius, 0, radius);
					matrix<FPT> row_tmp(row_padded_shape.rows, row_padded_shape.cols);
					matrix<FPT> row_tmp2(src.size1(), src.size2());
					correlate1d(row_tmp.data().begin() + radius, row_padded.data().begin(), row_padded.data().end(), weights2);
					crop(row_tmp2, row_tmp, 0, radius, 0, radius);

					matrix_shape col_padded_shape = {src.size2(), src.size1() + 2 * radius};
					matrix<FPT> col_padded(col_padded_shape.rows, col_padded_shape.cols);
					add_edge_padding(col_padded, trans(row_tmp2), 0, radius, 0, radius);
					matrix<FPT> col_tmp(col_padded_shape.rows, col_padded_shape.cols);
					matrix<FPT> col_tmp2(src.size2(), src.size1());
					correlate1d(col_tmp.data().begin() + radius, col_padded.data().begin(), col_padded.data().end(), weights0);
					crop(col_tmp2, col_tmp, 0, radius, 0, radius);

					dst += trans(col_tmp2);
				}
			}
		}
		
		static std::vector<matrix<FPT>>
		get_scale_space(
			const matrix<FPT> &image,
			const std::vector<FPT> &sigma)
		{
			matrix_shape image_shape = {image.size1(), image.size2()};
			size_t scales_num = sigma.size();
			std::vector<matrix<FPT>> scale_space(scales_num);

			for (ptrdiff_t k = 0; k < scales_num; k++) {
				const FPT magic_const = 3.0;
				scale_space[k] = matrix<FPT>(image_shape.rows, image_shape.cols);
				if(!USE_SCALING || sigma[k] < 5.0) {
					gaussian_laplace(scale_space[k], image, sigma[k]);
					scale_space[k] *= -pow(sigma[k], 2.0);
				} else {
					FPT coeff = magic_const / sigma[k];
					matrix_shape shape = {
						(size_t)((FPT)image_shape.rows * coeff),
						(size_t)((FPT)image_shape.cols * coeff)
					};
					matrix<FPT> scaled_image(shape.rows, shape.cols);
					resize_image(scaled_image, image);
					matrix<FPT> conv(shape.rows, shape.cols);
					gaussian_laplace(conv, scaled_image, magic_const);
					conv *= -pow(magic_const, 2.0);
					resize_image(scale_space[k], conv);
				}
			}
			return scale_space;
		}

		static FPT
		circle_overlap(FPT dist, FPT r1, FPT r2)
		{
			FPT ratio1 = (pow(dist, 2.0) + pow(r1, 2.0) - pow(r2, 2.0)) / (2.0 * dist * r1);
			ratio1 = std::min(std::max(ratio1, (FPT)-1.0), (FPT)1.0);
			FPT acos1 = acos(ratio1);
			FPT ratio2 = (pow(dist, 2.0) + pow(r2, 2.0) - pow(r1, 2.0)) / (2.0 * dist * r2);
			ratio2 = std::min(std::max(ratio2, (FPT)-1.0), (FPT)1.0);
			FPT acos2 = acos(ratio2);
			FPT a = -dist + r2 + r1;
			FPT b = dist - r2 + r1;
			FPT c = dist + r2 - r1;
			FPT d = dist + r2 + r1;
			FPT area = (pow(r1, 2.0) * acos1 + pow(r2, 2.0) * acos2 - 0.5 * sqrt(abs(a * b * c * d)));
			return area / (M_PI * pow(std::min(r1, r2), 2.0));
		}

		static float 
		blob_overlap(blob<FPT> blob1, blob<FPT> blob2)
		{
			FPT r1, r2, max_radius;
			if (abs(blob1.r) < EPS && abs(blob2.r) < EPS)
				return 0.0;
			else if (blob1.r > blob2.r) {
				max_radius = blob1.r;
				r1 = 1;
				r2 = blob2.r / blob1.r;
			} else {
				max_radius = blob2.r;
				r2 = 1;
				r1 = blob1.r / blob2.r;
			}
			FPT x1 = (FPT)blob1.x / max_radius;
			FPT y1 = (FPT)blob1.y / max_radius;
			FPT x2 = (FPT)blob2.x / max_radius;
			FPT y2 = (FPT)blob2.y / max_radius;
			FPT d = (FPT)sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0));
			if (d > r1 + r2) return 0.0;
			if (d <= abs(r1 - r2)) return 1.0;
			return circle_overlap(d, r1, r2);
		}

		/** \brief Структура, с которой умеет работать kdtree. */
		struct point {
			constexpr static const int DIM = 2;
			FPT x, y;
			FPT operator[] (int idx) const {
				switch (idx) {
				case 0:
					return x;
				case 1:
					return y;
				default:
					throw std::invalid_argument("Point operator[]: invalid argument!");
				}
			}
		};

	public:

		/**
		 * \brief 	Метод принимает массив пятен и возвращает такой поднабор, что никакие два
		 * 		из них не пересекаются более чем на overlap. Метод может быть полезен для
		 * 		объединения результатов нескольких вызовов detect_blobs(). 
		 * 
		 * \param blobs		Исходный массив, в котором окружности могут как угодно пересекаться.
		 * 
		 * \param overlap	Значение от 0 до 1. Если площадь пересечения двух объектов занимает
		 * 			от площади наименьшего объекта долю, большую чем overlap,
		 * 			то меньший объект удаляется.
		 * 
		 * \return Возвращает отфильтрованные объекты (некоторое подмножество @a blobs).
		 */
		static std::vector<blob<FPT>>
		prune_blobs(std::vector<blob<FPT>> blobs, FPT overlap)
		{
			if(blobs.empty()) return {};
			FPT sigma = -INF;
			for(auto &i : blobs) sigma = std::max(sigma, i.r);
			float distance = 2.0 * sigma * sqrt(2.0);
			std::vector<point> points(blobs.size());
			std::transform(blobs.cbegin(), blobs.cend(), points.begin(),
				[](const blob<FPT> &b) -> point { return {(FPT)b.x, (FPT)b.y}; });
			kdt::KDTree<point> tree(points);
			for(int i = 0; i < blobs.size(); i++) {
				point query = {(FPT)blobs[i].x, (FPT)blobs[i].y};
				auto neighbours = tree.radiusSearch(query, 2.0 * blobs[i].r);
				for(auto &j : neighbours) {
					if(i == j)
						continue;
					if(blob_overlap(blobs[i], blobs[j]) > overlap) {
						if (blobs[i].r > blobs[j].r)
							blobs[j].r = 0.0;
						else blobs[i].r = 0.0;
					}
				}
			}
			std::vector<blob<FPT>> res;
			std::copy_if (blobs.begin(), blobs.end(), std::back_inserter(res),
				[](const blob<FPT> &b){ return b.r > EPS; } );
			return res;
		}

	protected:
		constexpr static const FPT INF = 1e12;
		constexpr static const FPT EPS = 1e-12;
	};
};

#endif //BLOB_TRACKER_BLOB_DETECTOR_HPP
