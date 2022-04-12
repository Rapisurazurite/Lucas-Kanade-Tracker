//
// Created by Lazurite on 4/12/2022.
//
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cv_wrapper.h"
#include <sophus/se3.hpp>

namespace py = pybind11;
using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

//PYBIND11_MAKE_OPAQUE(VecVector2d);

extern void randomSamplePoint(const cv::Mat &left_img, cv::Mat &disparity_img, VecVector2d &pixels_ref,
                              vector<double> &depth_ref);

extern cv::Mat DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21);

py::tuple pyRandomSamplePoint(
        const py::array_t<uint8_t> &left_img,
        const py::array_t<uint8_t> &disparity_img) {
    cv::Mat left_img_cv = numpy_uint8_1c_to_cv_mat(left_img);
    cv::Mat disparity_img_cv = numpy_uint8_1c_to_cv_mat(disparity_img);
    VecVector2d pixels_ref;
    vector<double> depth_ref;
    randomSamplePoint(left_img_cv, disparity_img_cv, pixels_ref, depth_ref);
    return py::make_tuple(pixels_ref, depth_ref);
}

py::tuple pyDirectPoseEstimationMultiLayer(
        const py::array_t<uint8_t> &left_img,
        const py::array_t<uint8_t> &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref
        ){
    cv::Mat left_img_cv = numpy_uint8_1c_to_cv_mat(left_img);
    cv::Mat img2_cv = numpy_uint8_1c_to_cv_mat(img2);
    Sophus::SE3d T_cur_ref;
    cv::Mat imShow = DirectPoseEstimationMultiLayer(left_img_cv, img2_cv, px_ref, depth_ref, T_cur_ref);
    py::array_t<uint8_t> imShow_py = cv_mat_uint8_3c_to_numpy(imShow);
    return py::make_tuple(imShow_py, T_cur_ref);
}


void init_ex_directMethod(py::module &m){
    py::class_<Eigen::Vector2d>pyVector2d (m, "Vector2d");
    pyVector2d.def("__repr__", [](const Eigen::Vector2d &v) {
                return "<Vector2d x=" + std::to_string(v.x()) + ", y=" + std::to_string(v.y()) + ">";
            });
    pyVector2d.def_property_readonly("x", [](const Eigen::Vector2d &v) { return v.x(); });
    pyVector2d.def_property_readonly("y", [](const Eigen::Vector2d &v) { return v.y(); });
    py::class_<Sophus::SE3d> pySE3d (m, "SE3d");
    pySE3d.def("__repr__", [](const Sophus::SE3d &v) {
                return "<SE3d x=" + std::to_string(v.translation().x()) + ", y=" + std::to_string(v.translation().y()) + ", z=" + std::to_string(v.translation().z()) + ">";
            });
    m.def("randomSamplePoint", &pyRandomSamplePoint);
    m.def("directPoseEstimationMultiLayer", &pyDirectPoseEstimationMultiLayer);
}