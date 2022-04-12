//
// Created by Lazurite on 4/12/2022.
//

#ifndef CMAKE_EXAMPLE_CV_WRAPPER_H
#define CMAKE_EXAMPLE_CV_WRAPPER_H

namespace py = pybind11;

cv::Mat numpy_uint8_1c_to_cv_mat(const py::array_t<unsigned char> &input);

cv::Mat numpy_uint8_3c_to_cv_mat(const py::array_t<unsigned char> &input);

py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(const cv::Mat &input);

py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(const cv::Mat &input);

#endif //CMAKE_EXAMPLE_CV_WRAPPER_H
