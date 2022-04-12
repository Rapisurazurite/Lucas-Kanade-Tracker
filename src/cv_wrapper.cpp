//
// Created by Lazurite on 4/12/2022.
//

#include "opencv2/opencv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cv_wrapper.h"
/*
Python->C++ Mat
*/

cv::Mat numpy_uint8_1c_to_cv_mat(const py::array_t<unsigned char> &input) {

    if (input.ndim() != 2)
        throw std::runtime_error("1-channel image must be 2 dims ");

    py::buffer_info buf = input.request();

    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);

    return mat;
}


cv::Mat numpy_uint8_3c_to_cv_mat(const py::array_t<unsigned char> &input) {

    if (input.ndim() != 3)
        throw std::runtime_error("3-channel image must be 3 dims ");

    py::buffer_info buf = input.request();

    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

    return mat;
}


/*
C++ Mat ->numpy
*/
py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(const cv::Mat &input) {

    py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows,input.cols }, input.data);
    return dst;
}

py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(const cv::Mat &input) {

    py::array_t<unsigned char> dst = py::array_t<unsigned char>({ input.rows,input.cols,3}, input.data);
    return dst;
}

