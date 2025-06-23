#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../core/src/blob_detector/blob_detector.hpp"

namespace py = pybind11;
using namespace blobs;

matrix<double> numpy_to_matrix(py::array_t<double>& input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Expected 2D array (matrix)");
    
    auto rows = buf.shape[0];
    auto cols = buf.shape[1];
    matrix<double> result(rows, cols);
    auto ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = ptr[i * cols + j];
    return result;
}

void bind_blob(py::module& m) {
    py::class_<blob<double>>(m, "Blob")
        .def_readonly("x", &blob<double>::x)
        .def_readonly("y", &blob<double>::y)
        .def_readonly("r", &blob<double>::r);
}

void bind_detect_blobs(py::module& m) {
    m.def("detect_blobs",
        [](py::array_t<double>& image, double min_sigma, double max_sigma, size_t num_sigma,
           double threshold_abs, bool use_prune_blobs, double overlap,
           conv_method method, bool use_scaling) {
            auto matrix = numpy_to_matrix(image);
            switch(method) {
                case conv_method::AUTO:
                    if (use_scaling) {
                        return blob_detector<double, conv_method::AUTO, true>::detect_blobs(
                            matrix, min_sigma, max_sigma, num_sigma, 
                            threshold_abs, overlap, use_prune_blobs
                        );
                    } else {
                        return blob_detector<double, conv_method::AUTO, false>::detect_blobs(
                            matrix, min_sigma, max_sigma, num_sigma, 
                            threshold_abs, overlap, use_prune_blobs
                        );
                    }
                case conv_method::SIMPLE:
                    if (use_scaling) {
                        return blob_detector<double, conv_method::SIMPLE, true>::detect_blobs(
                            matrix, min_sigma, max_sigma, num_sigma, 
                            threshold_abs, overlap, use_prune_blobs
                        );
                    } else {
                        return blob_detector<double, conv_method::SIMPLE, false>::detect_blobs(
                            matrix, min_sigma, max_sigma, num_sigma, 
                            threshold_abs, overlap, use_prune_blobs
                        );
                    }
                case conv_method::FFT_SLOW:
                    if (use_scaling) {
                        return blob_detector<double, conv_method::FFT_SLOW, true>::detect_blobs(
                            matrix, min_sigma, max_sigma, num_sigma, 
                            threshold_abs, overlap, use_prune_blobs
                        );
                    } else {
                        return blob_detector<double, conv_method::FFT_SLOW, false>::detect_blobs(
                            matrix, min_sigma, max_sigma, num_sigma, 
                            threshold_abs, overlap, use_prune_blobs
                        );
                    }
                case conv_method::FFT_FAST:
                    if (use_scaling) {
                        return blob_detector<double, conv_method::FFT_FAST, true>::detect_blobs(
                            matrix, min_sigma, max_sigma, num_sigma, 
                            threshold_abs, overlap, use_prune_blobs
                        );
                    } else {
                        return blob_detector<double, conv_method::FFT_FAST, false>::detect_blobs(
                            matrix, min_sigma, max_sigma, num_sigma, 
                            threshold_abs, overlap, use_prune_blobs
                        );
                    }
                default:
                    throw std::runtime_error("Unsupported convolution method");
            }
        },
        py::arg("image"),
        py::arg("min_sigma"),
        py::arg("max_sigma"),
        py::arg("num_sigma"),
        py::arg("threshold_abs"),
        py::arg("use_prune_blobs") = false,
        py::arg("overlap") = 0.0,
        py::arg("method") = conv_method::AUTO,
        py::arg("use_scaling") = true,
        "Detect blobs using Laplacian of Gaussian\n\n"
        "Args:\n"
        "    image: 2D numpy array (grayscale image, dtype=float64)\n"
        "    min_sigma: Minimum sigma for LoG filter (double)\n"
        "    max_sigma: Maximum sigma for LoG filter (double)\n"
        "    num_sigma: Number of intermediate sigma values (int)\n"
        "    threshold_abs: Absolute intensity threshold (double)\n"
        "    use_prune_blobs: Whether to apply pruning (bool, default=False)\n"
        "    overlap: Maximum allowed overlap ratio [0-1] (double, default=0)\n"
        "    method: Convolution method (ConvMethod enum, default=AUTO)\n"
        "    use_scaling: Whether to use image scaling (bool, default=True)\n"
        "Returns:\n"
        "    List of detected Blob objects"
    );
}

void bind_prune_blobs(py::module& m) {
    m.def("prune_blobs",
        [](std::vector<blob<double>>& blobs, double overlap,
           conv_method method, bool use_scaling) {
                switch(method) {
                case conv_method::AUTO:
                    if (use_scaling) {
                        return blob_detector<double, conv_method::AUTO, true>::prune_blobs(
                            blobs, overlap
                        );
                    } else {
                        return blob_detector<double, conv_method::AUTO, false>::prune_blobs(
                            blobs, overlap
                        );
                    }
                case conv_method::SIMPLE:
                    if (use_scaling) {
                        return blob_detector<double, conv_method::SIMPLE, true>::prune_blobs(
                            blobs, overlap
                        );
                    } else {
                        return blob_detector<double, conv_method::SIMPLE, false>::prune_blobs(
                            blobs, overlap
                        );
                    }
                case conv_method::FFT_SLOW:
                    if (use_scaling) {
                        return blob_detector<double, conv_method::FFT_SLOW, true>::prune_blobs(
                            blobs, overlap
                        );
                    } else {
                        return blob_detector<double, conv_method::FFT_SLOW, false>::prune_blobs(
                            blobs, overlap
                        );
                    }
                case conv_method::FFT_FAST:
                    if (use_scaling) {
                        return blob_detector<double, conv_method::FFT_FAST, true>::prune_blobs(
                            blobs, overlap
                        );
                    } else {
                        return blob_detector<double, conv_method::FFT_FAST, false>::prune_blobs(
                            blobs, overlap
                        );
                    }
                default:
                    throw std::runtime_error("Unsupported convolution method");
            }
        },
        py::arg("blobs"),
        py::arg("overlap") = 0.0,
        py::arg("method") = conv_method::AUTO,
        py::arg("use_scaling") = true,
        "Filter overlapping blobs\n\n"
        "Args:\n"
        "    blobs: List of detected Blob objects\n"
        "    overlap: Maximum allowed overlap ratio [0-1] (double, default=0)\n"
        "    method: Convolution method (ConvMethod enum, default=AUTO)\n"
        "    use_scaling: Whether to use image scaling (bool, default=True)\n"
        "Returns:\n"
        "    List of non-overlapping Blob objects"
    );
}

PYBIND11_MODULE(_blob_internal, m) {
    py::enum_<conv_method>(m, "ConvMethod")
        .value("AUTO", conv_method::AUTO)
        .value("SIMPLE", conv_method::SIMPLE)
        .value("FFT_SLOW", conv_method::FFT_SLOW)
        .value("FFT_FAST", conv_method::FFT_FAST)
        .export_values();

    bind_blob(m);
    bind_detect_blobs(m);
    bind_prune_blobs(m);
}
