#ifndef BLOB_TRACKER_FFT_FAST_HPP
#define BLOB_TRACKER_FFT_FAST_HPP

#ifdef RASPBERRYPI
#include <arm_neon.h>
#endif

#include <vector>
#include <cmath>
#include <boost/pool/pool_alloc.hpp>

/**
 * \brief  Singleton class implementing the Fast Fourier Transform (FFT).
 *         Allows multiplying two polynomials of length n in O(n log n) time.
 * 
 * \tparam FPT     Floating point type (type used internally by the class as fpt).
 */
template<typename FPT>
struct fft_fast {
	fft_fast() = delete;

	/**
	 * \brief	The method initializes internal arrays used by compute(). Must
	 * 			be called at least once at the beginning. Calling compute()
	 *          before initialization is UB.
	 */
	static void 
	init()
	{
		prepPw();
		prepRoots();
		for(int i = 0; i <= maxBase; i++)
			prepRev(i, pw[i]);
	}

	struct complex_num {
		FPT x, y;
		complex_num() : x(0), y(0) {}
		complex_num(FPT xx, FPT yy) : x(xx), y(yy) {}
		explicit complex_num(FPT alp) : x(cosf(alp)), y(sinf(alp)) {}
		complex_num
		operator+(complex_num b) const {
			return {x + b.x, y + b.y};
		}
		complex_num
		operator-(complex_num b) const {
			return {x - b.x, y - b.y};
		}
		complex_num
		operator*(complex_num b) const {
			return {x * b.x - y * b.y, x * b.y + y * b.x};
		}
		complex_num
		conj() const {
			return {x, -y};
		}
	};

	/**
	 * \brief  Performs Fourier transform on the input array 'a'.
	 * 
	 * \param base  Logarithm of the size of input vectors 'a' and 'f' (base-2).
	 * \param a     Input array to apply Fourier transform to.
	 * 				Array size must be exactly 2^base.
	 * \param f     Output array to store the transform result.
	 * 				Array size must be exactly 2^base.
	 * 
	 * \return      Transform result (polynomial values at 2^base points).
	 */
	static void 
	compute(int base, const std::vector<complex_num> &a, std::vector<complex_num> &f)
	{
		size_t N = pw[base];
		for(ptrdiff_t i = 0; i < N; i++)
			f[i] = a[rev[base][i]];
		for(ptrdiff_t k = 1; k < N; k <<= 1) {
			for(ptrdiff_t i = 0; i < N; i += 2 * k) {
				/* ARM-NEON vectorization (см. https://developer.arm.com/architectures/instruction-sets
					/intrinsics/#f:@navigationhierarchiessimdisa=[Neon]) */
	#ifdef RASPBERRYPI
				ptrdiff_t R = k - k % 4;
				for(ptrdiff_t x = 0; x < R; x += 4) {
					float32x4x2_t a = vld2q_f32(reinterpret_cast<FPT *>(&f[y + x + k]));
					float32x4x2_t b = vld2q_f32(reinterpret_cast<FPT *>(&root[x + k]));
					float32x4x2_t c;
					c.val[0] = vmulq_f32(a.val[0], b.val[0]);
					c.val[1] = vmulq_f32(a.val[0], b.val[1]);
					c.val[0] = vmlsq_f32(c.val[0], a.val[1], b.val[1]);
					c.val[1] = vmlaq_f32(c.val[1], a.val[1], b.val[0]);
					a = vld2q_f32(reinterpret_cast<FPT *>(&f[y + x]));
					float32x4x2_t d;
					d.val[0] = vsubq_f32(a.val[0], c.val[0]);
					d.val[1] = vsubq_f32(a.val[1], c.val[1]);
					vst2q_f32(reinterpret_cast<FPT *>(&f[y + x + k]), d);
					d.val[0] = vaddq_f32(a.val[0], c.val[0]);
					d.val[1] = vaddq_f32(a.val[1], c.val[1]);
					vst2q_f32(reinterpret_cast<FPT *>(&f[y + x]), d);
				}
				for(ptrdiff_t x = R; x < k; x++) {
					complex_num z = f[y + x + k] * root[x + k];
					f[y + x + k] = f[y + x] - z;
					f[y + x] = f[y + x] + z;
				}
	#else //RASPBERRYPI
				for(ptrdiff_t j = 0; j < k; j++) {
					complex_num z = f[i + j + k] * root[j + k];
					f[i + j + k] = f[i + j] - z;
					f[i + j] = f[i + j] + z;
				}
	#endif //RASPBERRYPI
			}
		}
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
		while(N < n1 + n2) base++, N <<= 1;
		if(N > maxN)
			throw std::runtime_error("Multiply polynomials: length of input array > maxN");
		std::vector<complex_num> a(N);
		for(ptrdiff_t i = 0; i < N; i++) {
			FPT x = (i < n1) ? A[i] : 0.f;
			FPT y = (i < n2) ? B[i] : 0.f;
			a[i] = complex_num(x, y);
		}
		std::vector<complex_num> f(N);
		compute(base, a, f);
		for(int i = 0; i < N; i++) {
			int j = (N - i) & (N - 1);
			a[i] = (f[j] * f[j] - (f[i] * f[i]).conj()) * complex_num(0, -0.25 / (FPT)N);
		}
		compute(base, a, f);
		std::vector<FPT> C(n1 + n2 - 1);
		for(ptrdiff_t i = 0; i < n1 + n2 - 1; i++)
			C[i] = f[i].x;
		return C;
	}

private:
	static void 
	prepPw()
	{
		pw[0] = 1;
		for(ptrdiff_t k = 1; k <= maxBase; k++)
			pw[k] = pw[k - 1] << 1;
	}

	static void 
	prepRoots()
	{
		root[1] = complex_num(1, 0);
		for(ptrdiff_t k = 1; k < maxBase; ++k) {
			complex_num x(2 * PI / (FPT)pw[k + 1]);
			for(ptrdiff_t i = pw[k - 1]; i < pw[k]; ++i) {
				root[2 * i] = root[i];
				root[2 * i + 1] = root[i] * x;
			}
		}
	}

	static void 
	prepRev(int base, int N)
	{
		for(int i = 0; i < N; i++)
			rev[base][i] = (rev[base][i >> 1] >> 1) + ((i & 1) << (base - 1));
	}

private:
	static const int maxBase = 21;
	static const int maxN = 1 << maxBase;
	constexpr static const FPT PI = M_PI;
	static int pw[maxBase + 1];
	static complex_num root[maxN];
	static int rev[maxBase + 1][maxN];
};

template<typename FPT>
int fft_fast<FPT>::pw[maxBase + 1];
template<typename FPT>
typename fft_fast<FPT>::complex_num fft_fast<FPT>::root[maxN];
template<typename FPT>
int fft_fast<FPT>::rev[maxBase + 1][maxN];

#endif //BLOB_TRACKER_FFT_FAST_HPP