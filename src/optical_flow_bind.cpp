//
// Created by Lazurite on 4/10/2022.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cv_wrapper.h"

// this will make the vector exposed to python as a c++ object
// otherwise it will be a list
PYBIND11_MAKE_OPAQUE(std::vector<cv::KeyPoint>);
PYBIND11_MAKE_OPAQUE(std::vector<cv::Point2f>);

namespace py = pybind11;
using namespace std;


extern void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const vector<cv::KeyPoint> &kp1,
        vector<cv::KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false,
        bool has_initial = false
);

extern void OpticalFlowMultiLevel(
        const cv::Mat &img_1,
        const cv::Mat &img_2,
        const vector<cv::KeyPoint> &kp1,
        vector<cv::KeyPoint> &kp2_multiLevel,
        vector<bool> &success_multiLevel);

template<typename PointType, typename successType>
extern cv::Mat DrawOpticalFlowInImage(cv::Mat &img, vector<PointType> &kp1, vector<PointType> &kp2, vector<successType> &success);

extern cv::Mat DrawOpticalFlowInImage(
    cv::Mat &img,vector<cv::KeyPoint> &kp1,
    vector<cv::KeyPoint> &kp2,
    vector<bool> &success);

extern cv::Mat opticalFlowDetectUsingOpencv(
        cv::Mat &img_1,
        cv::Mat &img_2,
        vector<cv::KeyPoint> &kp1
        );

py::tuple pyOpticalFlowSingleLevel(
        py::array_t<uint8_t>& img1,
        py::array_t<uint8_t>& img2,
        const vector<cv::KeyPoint> &kp1,
        bool inverse = false,
        bool has_initial = false
        ){

    cv::Mat mat1 = numpy_uint8_1c_to_cv_mat(img1);
    cv::Mat mat2 = numpy_uint8_1c_to_cv_mat(img2);
    vector<bool> success_;
    vector<cv::KeyPoint> kp2;
    OpticalFlowSingleLevel(mat1, mat2, kp1, kp2, success_, inverse, has_initial);
    return py::make_tuple(success_, kp2);
}


py::tuple pyOpticalFlowDetectUsingOpencv(
        py::array_t<uint8_t>& img_1,
        py::array_t<uint8_t>& img_2,
        const vector<cv::KeyPoint> &kp1
        ){
    cv::Mat mat_1 = numpy_uint8_1c_to_cv_mat(img_1);
    cv::Mat mat_2 = numpy_uint8_1c_to_cv_mat(img_2);
    vector<cv::Point2f> pt1, pt2;
    for (auto& kp : kp1) {
        pt1.push_back(kp.pt);
    }
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(mat_1, mat_2, pt1, pt2, status, error);
    vector<bool> success;
    for (unsigned char &statu : status) {
        success.push_back(statu == 1);
    }
    return py::make_tuple(success, pt2);
}

py::tuple pyOpticalFlowMultiLevel(
        py::array_t<uint8_t>& img1,
        py::array_t<uint8_t>& img2,
        const vector<cv::KeyPoint> &kp1){
    cv::Mat mat1 = numpy_uint8_1c_to_cv_mat(img1);
    cv::Mat mat2 = numpy_uint8_1c_to_cv_mat(img2);
    vector<bool> success_;
    vector<cv::KeyPoint> kp2;
    OpticalFlowMultiLevel(mat1, mat2, kp1, kp2, success_);
    return py::make_tuple(success_, kp2);
}

template<typename T>
py::array_t<uint8_t> pyDrawOpticalFlowInImage(
        py::array_t<uint8_t>& img,
        vector<T> &kp1,
        vector<T> &kp2,
        vector<bool> &success
        ){
    cv::Mat mat = numpy_uint8_1c_to_cv_mat(img);
    cv::Mat result = DrawOpticalFlowInImage(mat, kp1, kp2, success);
    return cv_mat_uint8_3c_to_numpy(result);
}

template <typename T>
void print(T& t){
    py::print(t);
}


void init_ex_opticalFlow(py::module &m){
    py::class_<cv::Ptr<cv::GFTTDetector>>(m, "GFTTDetector");
    py::bind_vector<std::vector<cv::KeyPoint>>(m, "VectorKeyPoint");
    py::bind_vector<std::vector<cv::Point2f>>(m, "VectorPoint2f");
    py::class_<cv::KeyPoint>(m, "KeyPoint")
            .def("__repr__", [](const cv::KeyPoint& kp) {
                std::stringstream ss;
                ss << "KeyPoint(x=" << kp.pt.x << ", y=" << kp.pt.y << ")";
                return ss.str();
            });
    py::class_<cv::Point2f>(m, "Point2f")
            .def("__repr__", [](const cv::Point2f& pt) {
                std::stringstream ss;
                ss << "Point2f(x=" << pt.x << ", y=" << pt.y << ")";
                return ss.str();
            });
    m.def("detectKeypoint", [](py::array_t<u_int8_t>& img){
        cv::Mat mat = numpy_uint8_1c_to_cv_mat(img);
        std::vector<cv::KeyPoint> keypoints;
        cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);
        detector->detect(mat, keypoints);
        return keypoints;
    });
    m.def("OpticalFlowSingleLevel", &pyOpticalFlowSingleLevel, py::return_value_policy::take_ownership);
    m.def("OpticalFlowMultiLevel", &pyOpticalFlowMultiLevel, py::return_value_policy::take_ownership);
    m.def("opticalFlowDetectUsingOpencv", &pyOpticalFlowDetectUsingOpencv, py::return_value_policy::take_ownership);

    m.def("KeyPointToPoint2f", [](const vector<cv::KeyPoint> &kp){
        vector<cv::Point2f> pt;
        for (auto& kp_ : kp) {
            pt.push_back(kp_.pt);
        }
        return pt;
    });
//    m.def("DrawOpticalFlowInImage",
//          py::overload_cast<py::array_t<uint8_t>&, vector<cv::Point2f>&, vector<cv::Point2f>&, vector<bool>&>(&pyDrawOpticalFlowInImage<cv::Point2f>),
//                  py::return_value_policy::take_ownership);
//    m.def("DrawOpticalFlowInImage",
//          py::overload_cast<py::array_t<uint8_t>&, vector<cv::KeyPoint>&, vector<cv::KeyPoint>&, vector<bool>&>(&pyDrawOpticalFlowInImage<cv::KeyPoint>),
//          py::return_value_policy::take_ownership);
    m.def("DrawOpticalFlowInImage",
          py::overload_cast<py::array_t<uint8_t> &,
                  vector<cv::KeyPoint> &,
                  vector<cv::KeyPoint> &,
                  vector<bool> &>(&pyDrawOpticalFlowInImage<cv::KeyPoint>),
            py::return_value_policy::take_ownership);

    m.def("DrawOpticalFlowInImage",
          py::overload_cast<py::array_t<uint8_t> &,
                  vector<cv::Point2f> &,
                  vector<cv::Point2f> &,
                  vector<bool> &>(&pyDrawOpticalFlowInImage<cv::Point2f>),
            py::return_value_policy::take_ownership);


//    m.def("print", py::overload_cast<int &>(&print<int>));
//    m.def("print", py::overload_cast<std::string &>(&print<std::string>));
//
//    m.def("add_list", [](vector<int>& v, int x){
//        v.push_back(x);
//    });
//    m.def("add_list2", [](py::list v, int x){
//        v.append(x);
//    });
}