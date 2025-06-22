#ifndef BLOB_TRACKER_FFT_FAST_HPP
#define BLOB_TRACKER_FFT_FAST_HPP

#ifdef RASPBERRYPI
#include <arm_neon.h>
#endif

#include <vector>
#include <cmath>
#include <boost/pool/pool_alloc.hpp>

/**
 * \brief 	Singleton-класс, реализующий быстрое преобразование Фурье.
 * 			Позволяет перемножить два многочлена длины n за O(nlogn).
 * 
 * \tparam FPT 		floating point type (тип, используемый внутри класса в качестве fpt).
 */
template<typename FPT>
struct fft_fast {
	fft_fast() = delete;

	/**
	 * \brief 	Метод ининциализирует внутренние массивы, которые используются compute().
	 * 			Должен быть вызван как мимнимум один раз в начале. Если Вы вызовете
	 * 			compute до инициализации, то получите неопределенное поведение.
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
	 * \brief 	Метод совершает преобразование Фурье над исходным массивом a.
	 * 
	 * \param base	Логарифм от размера входных векторов a и f.
	 * \param a		Входной массив, к которому требуется применить преобразование Фурье.
	 * 				Размер массива должен быть в точности равен 2^base.
	 * \param f		Выходной массив, в который требуется сохранить результат преобразования.
	 * 				Размер массива должен быть в точности равен 2^base.
	 * 
	 * \return		Результат преобразования (значния многочлена в 2^base точках).
	 */	
	static void 
	compute(int base, const std::vector<complex_num> &a, std::vector<complex_num> &f)
	{
		size_t N = pw[base];
		for(ptrdiff_t i = 0; i < N; i++)
			f[i] = a[rev[base][i]];
		for(ptrdiff_t k = 1; k < N; k <<= 1) {
			for(ptrdiff_t i = 0; i < N; i += 2 * k) {
				/* ARM-NEON векторизация (см. https://developer.arm.com/architectures/instruction-sets
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
	 * \brief 	Метод перемножает два многочлена при помощи преобразования Фурье за O(nlogn).
	 * 
	 * \param A		Первый многочлен. A[y] соответствует коэффициенту при x^y.
	 * \param B		Второй многочлен. B[y] соответствует коэффициенту при x^y.
	 * 
	 * \return		Произведение многочленов A и B.
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