//
// Created by Lazurite on 4/9/2022.
//
#include <iostream>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>
#include "imageBasic.h"

using namespace std;

cv::Mat imageFromPython(py::array_t<uint8_t> &img){
    auto rows = img.shape(0);
    auto cols = img.shape(1);
    auto type = CV_8UC3;

    cv::Mat img2(rows, cols, type, (unsigned char*)img.data());
    return img2;
}

void imshow(cv::Mat &img){
    cv::imshow("image", img);
    cv::waitKey(0);
}

void init_ex_imageBasic(py::module &m){
    py::class_<cv::Mat> mat(m, "Mat");
    m.def("imageFromPython", &imageFromPython);
    m.def("imshow", &imshow, py::arg("img"));
}