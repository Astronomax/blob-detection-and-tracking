#ifndef BLOB_TRACKER_FFT_SLOW_HPP
#define BLOB_TRACKER_FFT_SLOW_HPP

#include <vector>
#include <complex>
#include <boost/pool/pool_alloc.hpp>

/**
 * \brief  Singleton class implementing the Fast Fourier Transform (FFT).
 *         Allows multiplying two polynomials of length n in O(n log n) time.
 * 
 * \tparam FPT     Floating point type (type used internally by the class as fpt).
 */
template<typename FPT>
struct fft_slow {
	fft_slow() = delete;

	typedef std::complex<FPT> P;

	/**
	 * \brief  Performs Fourier transform on the input array a.
	 * 
	 * \param n         Length of input array a. Must be exactly a power of 2.
	 * \param a         Input array for Fourier transform.
	 *                  Array size must satisfy n = 2^base for some integer base.
	 * \param inverse   Transform direction:
	 *                  true  -> inverse transform
	 *                  false -> forward transform
	 * 
	 * \return      Transform result (polynomial values at 2^base points).
	 */
	static void
	compute(int n, std::vector<P> &a, bool inverse)
	{
		if (n == 1) return;

		for (int i = 1, j = 0; i < n; i++) {
			int bit = n >> 1;
			for (; j >= bit; bit >>= 1) j -= bit;
			j += bit;
			if (i < j) std::swap(a[i], a[j]);
		}

		for (int i = 2; i <= n; i <<= 1) {
			for (int j = 0; j < n; j += i) {
				P w = 1, w0 = std::polar<FPT>(1.0, 2.0 * M_PI / i);
				if (inverse) w0 = std::polar<FPT>(1.0, -2.0 * M_PI / i);
				for (int k = 0; k < (i >> 1); k++) {
					P f = a[j + k] + w * a[j + (i >> 1) + k];
					P g = a[j + k] - w * a[j + (i >> 1) + k];
					a[j + k] = f;
					a[j + (i >> 1) + k] = g;
					w *= w0;
				}
			}
		}
		if (inverse)
			for (int i = 0; i < n; ++i) a[i] /= n;
	}

	/**
	 * \brief  Multiplies two polynomials using Fourier transform in O(n log n) time.
	 * 
	 * \param A     First polynomial. A[y] corresponds to the coefficient of x^y.
	 * \param B     Second polynomial. B[y] corresponds to the coefficient of x^y.
	 * 
	 * \return      Product of polynomials A and B.
	 */
	static std::vector<FPT>
	multiply_polynomials(const std::vector<FPT> &A, const std::vector<FPT> &B)
	{
		size_t n1 = A.size(), n2 = B.size();
		int base = 1, N = 2;
		while(N < std::max(n1, n2)) base++, N <<= 1;
		base++; N <<= 1;

		std::vector<P> complex_a(N), complex_b(N);
		for(int i = 0; i < A.size(); i++) complex_a[i] = {A[i], 0.0};
		for(int i = 0; i < B.size(); i++) complex_b[i] = {B[i], 0.0};

		compute(N, complex_a, false);
		compute(N, complex_b, false);
		for (int i = 0; i < N; i++)
			complex_a[i] *= complex_b[i];
		compute(N, complex_a, true);
		std::vector<FPT> res(n1 + n2 - 1);
		for (int i = 0; i < res.size(); i++)
			res[i] = complex_a[i].real();
		return res;
	}
};

#endif //BLOB_TRACKER_FFT_SLOW_HPP