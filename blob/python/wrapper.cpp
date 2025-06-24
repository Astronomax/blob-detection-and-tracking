#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../core/src/blob_detector/blob_detector.hpp"
#include "../core/src/blob_tracker/blob_tracker.hpp"

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
        .def_readwrite("x", &blob<double>::x)
        .def_readwrite("y", &blob<double>::y)
        .def_readwrite("r", &blob<double>::r);
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

void bind_blob_tracker(py::module& m) {
    py::class_<blob_tracker>(m, "BlobTracker")
        .def(py::init<bool, bool, int>(),
            py::arg("use_prediction") = true,
            py::arg("use_ttl") = true,
            py::arg("ttl") = 10,
            "Initialize blob tracker\n\n"
            "Args:\n"
            "    use_prediction: Whether to use Kalman filter prediction (default: True)\n"
            "    use_ttl: Whether to use Time-To-Live for objects (default: True)\n"
            "    ttl: Number of frames to keep objects after they disappear (default: 10)"
        )
        .def("track", 
            [](blob_tracker& tracker, 
               const std::vector<blob<double>>& blobs,
               float move_threshold,
               float scale_threshold) {
                
                auto tracked_objects = tracker.track(blobs, move_threshold, scale_threshold);

                std::vector<py::dict> result;
                for (const auto& obj : tracked_objects) {
                    py::dict py_obj;
                    py_obj["id"] = obj.id;
                    py_obj["status"] = static_cast<int>(obj.status);
                    py_obj["x"] = obj.blob_data.x;
                    py_obj["y"] = obj.blob_data.y;
                    py_obj["r"] = obj.blob_data.r;
                    result.push_back(py_obj);
                }
                return result;
            },
            py::arg("blobs"),
            py::arg("move_threshold") = std::numeric_limits<float>::infinity(),
            py::arg("scale_threshold") = 1.5f,
            "Track blobs between frames\n\n"
            "Args:\n"
            "    blobs: List of detected Blob objects\n"
            "    move_threshold: Max allowed movement between frames (pixels)\n"
            "    scale_threshold: Max allowed scale change (ratio)\n"
            "Returns:\n"
            "    List of dicts with tracking info (id, status, x, y, r)"
        );
}

PYBIND11_MODULE(_blob_internal, m) {
    py::enum_<conv_method>(m, "ConvMethod")
        .value("AUTO", conv_method::AUTO)
        .value("SIMPLE", conv_method::SIMPLE)
        .value("FFT_SLOW", conv_method::FFT_SLOW)
        .value("FFT_FAST", conv_method::FFT_FAST)
        .export_values();

    py::enum_<blob_tracker::o_status>(m, "ObjectStatus")
        .value("BORN", blob_tracker::o_status::BORN)
        .value("ALIVE", blob_tracker::o_status::ALIVE)
        .value("GHOST", blob_tracker::o_status::GHOST)
        .value("DIED", blob_tracker::o_status::DIED)
        .export_values();

    bind_blob(m);
    bind_detect_blobs(m);
    bind_prune_blobs(m);
    bind_blob_tracker(m);
}
