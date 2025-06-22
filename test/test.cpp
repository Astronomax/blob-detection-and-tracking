#include <gtest/gtest.h>
#include "../blob_detector/blob_detector.hpp"
#include "../blob_tracker/blob_tracker.hpp"
#include <random>

namespace blobs {
    template<typename FPT = float, blobs::conv_method CONV_METHOD = conv_method::AUTO, bool USE_SCALING = true>
    class blob_detector_unit_test : blobs::blob_detector<FPT, CONV_METHOD, USE_SCALING> {
        FRIEND_TEST(blob_detector_unit_test, correlate1d_test1_float);
        FRIEND_TEST(blob_detector_unit_test, correlate1d_test1_double);
        FRIEND_TEST(blob_detector_unit_test, correlate1d_test2);
        FRIEND_TEST(blob_detector_unit_test, correlate1d_test3);
        FRIEND_TEST(blob_detector_unit_test, correlate1d_test4);
        FRIEND_TEST(blob_detector_unit_test, correlate1d_test5);
        FRIEND_TEST(blob_detector_unit_test, gaussian_filter1d_float);
        FRIEND_TEST(blob_detector_unit_test, gaussian_filter1d_double);
        FRIEND_TEST(blob_detector_unit_test, gaussian_laplace_test1);
        FRIEND_TEST(blob_detector_unit_test, gaussian_laplace_alt_impl);
        FRIEND_TEST(blob_detector_unit_test, maximum_filter1d);
        FRIEND_TEST(blob_detector_unit_test, maximum_filter1d_test2);
    };

    std::vector<double> multiply_polynomials(
            const std::vector<double> &a,
            const std::vector<double> &b) {
        std::vector<double> c(a.size() + b.size());
        for(int i = 0; i < a.size(); i++)
            for(int j = 0; j < b.size(); j++)
                c[i + j] += a[i] * b[j];
        return c;
    }

    template<typename ITER1, typename ITER2>
    static void correlate1d_edaeda(ITER1 beg_dst, ITER2 beg_src, ITER2 end_src, const std::vector<double> &fw) {
        size_t n = end_src - beg_src;
        std::vector<double> a(n + 2 * (fw.size() / 2));
        std::copy(beg_src, end_src, a.begin() + fw.size() / 2);
        for(int i = 0; i < fw.size() / 2; i++) a[i] = *beg_src;
        for(int i = 0; i < fw.size() / 2; i++) a[i + fw.size() / 2 + n] = *(end_src - 1);
        std::vector<double> c(a.size() + fw.size());
        c = multiply_polynomials(a, fw);
        std::copy(c.begin() + fw.size() - 1, c.begin() + fw.size() - 1 + n, beg_dst);
    }

    TEST(blob_detector_unit_test, correlate1d_test1_float) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<float> uniform(0, 1);
        fft_fast<float>::init();
        std::vector<float> src(100);
        for(auto &i : src) i = uniform(random_engine);
        std::vector<float> dst(100);
        std::vector<float> dst1(100);
        std::vector<float> fw(30);
        for(auto &i : fw) i = uniform(random_engine);
        blob_detector_unit_test<float, blobs::conv_method::SIMPLE>::correlate1d_nearest(dst.begin(), src.begin(), src.end(), fw);
        std::reverse(fw.begin(), fw.end());
        blob_detector_unit_test<float, blobs::conv_method::FFT_FAST>::correlate1d_nearest(dst1.begin(), src.begin(), src.end(), fw);
        for(int i = 0; i < 100; i++)
            ASSERT_NEAR(dst[i], dst1[i], 1e-5);
    }

    TEST(blob_detector_unit_test, correlate1d_test1_double) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<double> uniform(0, 1);
        fft_fast<double>::init();
        std::vector<double> src(100);
        for(auto &i : src) i = uniform(random_engine);
        std::vector<double> dst(100);
        std::vector<double> dst1(100);
        std::vector<double> fw(30);
        for(auto &i : fw) i = uniform(random_engine);
        blob_detector_unit_test<double, blobs::conv_method::SIMPLE>::correlate1d_nearest(dst.begin(), src.begin(), src.end(), fw);
        std::reverse(fw.begin(), fw.end());
        correlate1d_edaeda(dst1.begin(), src.begin(), src.end(), fw);
        for(int i = 0; i < 100; i++)
            ASSERT_NEAR(dst[i], dst1[i], 1e-7);
    }

    TEST(blob_detector_unit_test, correlate1d_test2) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<double> uniform(0, 1);
        fft_fast<double>::init();
        std::vector<double> src(100);
        for(auto &i : src) i = uniform(random_engine);
        std::vector<double> dst(100);
        std::vector<double> dst1(100);
        std::vector<double> fw(30);
        for(auto &i : fw) i = uniform(random_engine);
        blob_detector_unit_test<double, blobs::conv_method::SIMPLE>::correlate1d_nearest(dst.begin(), src.begin(), src.end(), fw);
        std::reverse(fw.begin(), fw.end());
        blob_detector_unit_test<double, blobs::conv_method::FFT_FAST>::correlate1d_nearest(dst1.begin(), src.begin(), src.end(), fw);
        for(int i = 0; i < 100; i++)
            ASSERT_NEAR(dst[i], dst1[i], 1e-4);
    }

    TEST(blob_detector_unit_test, correlate1d_test3) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<float> uniform(0, 1);
        fft_fast<float>::init();
        std::vector<float> src(10000);
        for(int i = 0; src.begin() + i != src.end(); i++)
            *(src.begin() + i) = uniform(random_engine);
        std::vector<float> dst(10000);
        std::vector<float> dst1(10000);
        std::vector<float> fw(30);
        for(auto &i : fw) i = uniform(random_engine);
        blob_detector_unit_test<float, blobs::conv_method::SIMPLE>::correlate1d_nearest(dst.begin(), src.begin(), src.end(), fw);
        std::reverse(fw.begin(), fw.end());
        blob_detector_unit_test<float, blobs::conv_method::FFT_FAST>::correlate1d_nearest(dst1.begin(), src.begin(), src.end(), fw);
        for(int i = 0; i < 10000; i++)
            ASSERT_NEAR(dst[i], dst1[i], 1e-3);
    }

    TEST(blob_detector_unit_test, correlate1d_test4) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<float> uniform(0, 1);
        fft_fast<float>::init();
        std::vector<float> src(10000);
        for(int i = 0; src.begin() + i != src.end(); i++)
            *(src.begin() + i) = uniform(random_engine);
        std::vector<float> dst(10000);
        std::vector<float> dst1(10000);
        std::vector<float> fw(15, 1.0);
        blob_detector_unit_test<float, blobs::conv_method::FFT_FAST>::correlate1d_nearest(dst.begin(), src.begin(), src.end(), fw);

        std::vector<float> src2(src.size() + 2 * (fw.size() / 2));
        std::copy(src.begin(), src.end(), src2.begin() + fw.size() / 2);
        for(int i = 0; i < fw.size() / 2; i++) src2[i] = *src.begin();
        for(int i = 0; i < fw.size() / 2; i++) src2[i + fw.size() / 2 + src.size()] = *(src.end() - 1);

        blob_detector_unit_test<float, blobs::conv_method::FFT_FAST>::correlate1d(dst1.begin(), src2.begin(), src2.end(), fw);
        for(int i = 0; i < 10000; i++)
            ASSERT_NEAR(dst[i], dst1[i], 1e-5);
    }

    TEST(blob_detector_unit_test, correlate1d_test5) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<float> uniform(0, 1);
        fft_fast<float>::init();
        std::vector<float> src(100);
        for(int i = 0; src.begin() + i != src.end(); i++)
            *(src.begin() + i) = uniform(random_engine);
        std::vector<float> dst(100);
        std::vector<float> dst1(100);
        std::vector<float> fw(15, 1.0);
        blob_detector_unit_test<float, blobs::conv_method::FFT_FAST>::correlate1d_nearest(dst.begin(), src.begin(), src.end(), fw);

        std::vector<float> src2(src.size() + 2 * (fw.size() / 2));
        std::copy(src.begin(), src.end(), src2.begin() + fw.size() / 2);
        for(int i = 0; i < fw.size() / 2; i++) src2[i] = *src.begin();
        for(int i = 0; i < fw.size() / 2; i++) src2[i + fw.size() / 2 + src.size()] = *(src.end() - 1);

        blob_detector_unit_test<float, blobs::conv_method::SIMPLE>::correlate1d(dst1.begin(), src2.begin(), src2.end(), fw);
        for(int i = 0; i < 100; i++)
            ASSERT_NEAR(dst[i], dst1[i], 1e-5);
    }

    TEST(blob_detector_unit_test, gaussian_filter1d_float) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<float> uniform(0, 1);
        fft_fast<float>::init();
        std::vector<float> src(100);
        for(auto &i : src) i = uniform(random_engine);
        std::vector<float> dst(100);
        std::vector<float> dst1(100);
        blob_detector_unit_test<float, blobs::conv_method::SIMPLE>::gaussian_filter1d(dst.begin(), src.begin(), src.end(), 5.f, 2);
        blob_detector_unit_test<float, blobs::conv_method::FFT_FAST>::gaussian_filter1d(dst1.begin(), src.begin(), src.end(), 5.f, 2);
        for(int i = 0; i < 100; i++)
            ASSERT_NEAR(dst[i], dst1[i], 1e-5);
    }

    TEST(blob_detector_unit_test, gaussian_filter1d_double) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<float> uniform(0, 1);
        fft_fast<double>::init();
        std::vector<double> src(100);
        for(auto &i : src) i = uniform(random_engine);
        std::vector<float> dst(100);
        std::vector<float> dst1(100);
        blob_detector_unit_test<double, blobs::conv_method::SIMPLE>::gaussian_filter1d(dst.begin(), src.begin(), src.end(), 1.f, 2);
        blob_detector_unit_test<double, blobs::conv_method::FFT_FAST>::gaussian_filter1d(dst1.begin(), src.begin(), src.end(), 1.f, 2);
        for(int i = 0; i < 100; i++)
            ASSERT_NEAR(dst[i], dst1[i], 0.02);
    }

    TEST(blob_detector_unit_test, gaussian_laplace_test1) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<float> uniform(0, 1);
        fft_fast<float>::init();
        matrix<float> src(100, 100);
        for(int i = 0; src.data().begin() + i != src.data().end(); i++)
            *(src.data().begin() + i) = uniform(random_engine);
        matrix<float> dst(100, 100);
        matrix<float> dst1(100, 100);
        blob_detector_unit_test<float, blobs::conv_method::SIMPLE>::gaussian_laplace(dst, src, 1.f);
        blob_detector_unit_test<float, blobs::conv_method::FFT_FAST>::gaussian_laplace(dst1, src, 1.f);
        for(int i = 0; i < 100; i++)
            for(int j = 0; j < 100; j++)
                ASSERT_NEAR(dst(i, j), dst1(i, j), 1e-5);
    }

    TEST(blob_detector_unit_test, gaussian_laplace_alt_impl) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<double> uniform(0, 1);
        fft_fast<double>::init();
        matrix<double> src(3, 3);
        for(int i = 0; src.data().begin() + i != src.data().end(); i++)
            *(src.data().begin() + i) = uniform(random_engine);
        matrix<double> dst(3, 3);
        matrix<double> dst1(3, 3);
        blob_detector_unit_test<double, blobs::conv_method::SIMPLE>::gaussian_laplace(dst, src, 1.f);
        blob_detector_unit_test<double, blobs::conv_method::FFT_FAST>::gaussian_laplace_alt_impl(dst1, src, 1.f);
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                ASSERT_NEAR(dst(i, j), dst1(i, j), 1e-5);
    }

    TEST(Multiply, test1) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<float> uniform(0, 1);
        fft_fast<double>::init();
        std::vector<double> a(100);
        for(auto &i : a) i = uniform(random_engine);
        std::vector<double> b(30);
        for(auto &i : b) i = uniform(random_engine);
        auto c1 = fft_slow<double>::multiply_polynomials(a, b);
        auto c2 = fft_fast<double>::multiply_polynomials(a, b);//, a.size(), b.size());
        for(int i = 0; i < a.size() + b.size() - 1; i++)
            ASSERT_NEAR(c1[i], c2[i], 1e-5);
    }

    TEST(Multiply, test2) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<double> uniform(0, 1);
        fft_fast<double>::init();
        std::vector<double> a(10000);
        for(auto &i : a) i = uniform(random_engine);
        std::vector<double> b(100);
        for(auto &i : b) i = uniform(random_engine);
        auto c1 = fft_slow<double>::multiply_polynomials(a, b);
        auto c2 = multiply_polynomials(a, b);
        for(int i = 0; i < a.size() + b.size() - 1; i++)
            ASSERT_NEAR(c1[i], c2[i], 1e-7);
    }

    std::vector<float> maximum_filter1d(const std::vector<float> &a, size_t filter_size) {
        std::vector<float> res(a.size());
        for(ptrdiff_t i = 0; i < a.size(); i++) {
            res[i] = -1e18;
            for(ptrdiff_t j = 0; j < filter_size; j++) {
                ptrdiff_t ind = i + j - (ptrdiff_t)filter_size / 2;
                if(ind >= 0 && ind < a.size())
                    res[i] = std::max(res[i], a[ind]);
            }
        }
        return res;
    }

    TEST(blob_detector_unit_test, maximum_filter1d) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<double> uniform(0, 1);
        std::vector<float> a(100);
        for(auto &i : a) i = uniform(random_engine);
        std::vector<float> c1(100);
        blob_detector_unit_test<float>::maximum_filter1d(c1.begin(), a.begin(), a.end(), 3);
        auto c2 = maximum_filter1d(a, 3);
        for(int i = 0; i < a.size(); i++)
            ASSERT_NEAR(c1[i], c2[i], 1e-7);
    }

    TEST(blob_detector_unit_test, maximum_filter1d_test2) {
        std::random_device random_device;
        std::default_random_engine random_engine(random_device());
        std::uniform_real_distribution<float> uniform(0, 1);
        std::vector<float> a(100);
        for(auto &i : a) i = uniform(random_engine);
        std::vector<float> c1(100);
        blob_detector_unit_test<float>::maximum_filter1d(c1.begin(), a.begin(), a.end(), 10);
        auto c2 = maximum_filter1d(a, 10);
        for(int i = 0; i < a.size(); i++)
            ASSERT_NEAR(c1[i], c2[i], 1e-7);
    }

    TEST(blob_tracker, test1) {
        blob_tracker tracker(true);
        std::vector<blob<float>> blobs = {{0, 0, 1}};
        auto objects = tracker.track(blobs, 10.f, 1.7);
        ASSERT_EQ(objects.size(), 1);
        ASSERT_EQ(objects[0].status, blob_tracker::o_status::BORN);
        blobs = {{1000, 0, 1}};

        objects = tracker.track(blobs, 100.f, 1.7);
        ASSERT_EQ(objects.size(), 2);
        for(auto &o : objects) {
            if(o.blob_data.y == 0)
                ASSERT_EQ(o.status, blob_tracker::o_status::DIED);
            else ASSERT_EQ(o.status, blob_tracker::o_status::BORN);
        }
        for(int i = 0; i < 10; i++) {
            objects = tracker.track(blobs, 100.f, 1.7);
            ASSERT_EQ(objects.size(), 1);
            ASSERT_EQ(objects[0].status, blob_tracker::o_status::ALIVE);
        }
    }

    TEST(blob_tracker, test2) {
        blob_tracker tracker(true);
        std::vector<blob<float>> blobs = {{0, 0, 1}, {0, 1, 1}, {0, 2, 1}};
        auto objects = tracker.track(blobs, 1e12, 1.7);
        ASSERT_EQ(objects.size(), 3);
        for(auto &o : objects)
            ASSERT_EQ(o.status, blob_tracker::o_status::BORN);
        blobs = {{0, 4, 1}, {0, 3, 1}, {0, 5, 1}};
        //auto objects1 = tracker.track(blobs, 1e12, 1.7);
        auto objects1 = tracker.track(blobs, 100.f, 1.7);
        ASSERT_EQ(objects.size(), 3);

        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                if(objects[i].id == objects1[j].id)
                    ASSERT_NEAR(objects[i].blob_data.x + 3.0, objects1[j].blob_data.x, 1e-5);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}