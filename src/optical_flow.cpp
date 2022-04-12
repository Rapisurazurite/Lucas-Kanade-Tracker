//
// Created by Lazurite on 4/9/2022.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "optical_flow.h"

using namespace std;
using namespace cv;

string file_1 = "../res/LK1.png";
string file_2 = "../res/LK2.png";


/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
           + xx * (1 - yy) * img.at<uchar>(y, x_a1)
           + (1 - xx) * yy * img.at<uchar>(y_a1, x)
           + xx * yy * img.at<uchar>(y_a1, x_a1);
}


/**
 * @brief calculateOpticalFlow - calculate optical flow for each keypoint in range
 * @param [in] range of keypoints
 */
void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range) {
    // parameters
    const int half_patch_size = 4;
    const int iterations = 10;
    // run for each keypoint
//    cout << "range: " << range.start << " " << range.end << endl;
    for (size_t i = range.start; i < range.end; i++) {
        cv::KeyPoint kp = kp1[i];
        double dx = 0, dy = 0;
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }
        double cost = 0, lastCost = 0;
        bool succ = true; // indicate whether the keypoint is successfully tracked

        // Gauss-Newton iteration
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J = Eigen::Vector2d::Zero();

        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Vector2d update;
            cost = calBatchUpdate(half_patch_size, kp, dx, dy, J, H, b, update, iter);
            if (std::isnan(update[0])) {
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                break;
            }
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
            if (update.norm() < 1e-2) {
                break;
            }
        }
        success[i] = succ;
        // set kp2
        kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}


double OpticalFlowTracker::calBatchUpdate(const int half_patch_size, const cv::KeyPoint &kp, double dx, double dy,
                                          Eigen::Vector2d &J, Eigen::Matrix2d &H, Eigen::Vector2d &b,
                                          Eigen::Vector2d &update, int iter) {
    double cost = 0;

    if (!inverse){
        H = Eigen::Matrix2d::Zero();
        b = Eigen::Vector2d::Zero();
    } else {
        b = Eigen::Vector2d::Zero();
    }

    for (int x = -half_patch_size; x < half_patch_size; x++) {
        for (int y = -half_patch_size; y < half_patch_size; y++) {
            double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                           GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
            if (inverse == false){
                J = -1.0 * getDxDyOfImage(img2, cv::Point2f(kp.pt.x + x + dx, kp.pt.y + y + dy));
            } else if (iter == 0){
                J = -1.0 * getDxDyOfImage(img1, cv::Point2f(kp.pt.x + x, kp.pt.y + y));
            }
            b += -error * J;
            cost += error * error;
            if (!inverse or iter == 0){
                H += J * J.transpose();
            }
        }
    }
    update= H.ldlt().solve(b);
    return cost;
}

Eigen::Vector2d OpticalFlowTracker::getDxDyOfImage(const cv::Mat &img, const cv::Point2f &p) {
    double dx, dy;
    dx = 0.5 * (GetPixelValue(img, p.x + 1, p.y) - GetPixelValue(img, p.x - 1, p.y));
    dy = 0.5 * (GetPixelValue(img, p.x, p.y + 1) - GetPixelValue(img, p.x, p.y - 1));
    return Eigen::Vector2d{dx, dy};
}



void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const vector<cv::KeyPoint> &kp1,
        vector<cv::KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse,
        bool has_initial
) {
    // resize the shape of vector
    kp2.resize(kp1.size());
    success.resize(kp1.size());

    // use opencv to parallelize the computation
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    cv::parallel_for_(
            cv::Range(0, kp1.size()),
            std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1)
    );
}


void OpticalFlowMultiLevel(
        const cv::Mat &img_1,
        const cv::Mat &img_2,
        const vector<cv::KeyPoint> &kp1,
        vector<cv::KeyPoint> &kp2_multiLevel,
        vector<bool> &success_multiLevel){

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scale[] = {1.0, 0.5, 0.25, 0.125};
    //create pyramid
    vector<cv::Mat> pyr1, pyr2;
    pyr1.push_back(img_1);
    pyr2.push_back(img_2);
    for (int i = 1; i < pyramids; i++) {
        cv::Mat tmp1, tmp2;
        cv::pyrDown(pyr1[i-1], tmp1);
        cv::pyrDown(pyr2[i-1], tmp2);
        pyr1.push_back(tmp1);
        pyr2.push_back(tmp2);
    }

    vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp : kp1){
        cv::KeyPoint kp_top = kp;
        kp_top.pt *= scale[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--){
        success_multiLevel.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success_multiLevel, true, true);
        if (level > 0){
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }
    for (auto &kp: kp2_pyr)
        kp2_multiLevel.push_back(kp);
}


template cv::Mat DrawOpticalFlowInImage(cv::Mat &img, vector<cv::Point2f> &kp1, vector<cv::Point2f> &kp2, vector<bool> &success);
template cv::Mat DrawOpticalFlowInImage(cv::Mat &img, vector<cv::Point2f> &kp1, vector<cv::Point2f> &kp2, vector<uchar> &success);
cv::Mat DrawOpticalFlowInImage(cv::Mat &img, vector<cv::KeyPoint> &kp1, vector<cv::KeyPoint> &kp2, vector<bool> &success) {
    vector<cv::Point> p1, p2;
    for (size_t i = 0; i < kp1.size(); i++) {
            p1.push_back(kp1[i].pt);
            p2.push_back(kp2[i].pt);
        }
    return DrawOpticalFlowInImage(img, p1, p2, success);
}


cv::Mat opticalFlowDetectUsingOpencv(cv::Mat &img_1, cv::Mat &img_2, vector<cv::KeyPoint> &kp1) {
    vector<cv::Point2f> pt1, pt2;
    for (auto &kp: kp1) {
        pt1.push_back(kp.pt);
    }
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img_1, img_2, pt1, pt2, status, error);
    return DrawOpticalFlowInImage(img_2, pt1, pt2, status);
}

#ifdef MAIN_OPTICAL_FLOW
int main() {
    // Read images
    cv::Mat img_1 = imread(file_1, cv::IMREAD_GRAYSCALE);
    cv::Mat img_2 = imread(file_2, cv::IMREAD_GRAYSCALE);

    // Assert that the images are valid
    if (img_1.empty() || img_2.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Find key points in the images, using GFTT algorithm
    vector<cv::KeyPoint> kp1;
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);
    detector->detect(img_1, kp1);

    vector<cv::KeyPoint> kp2_singleLevel;
    vector<bool> success_singleLevel;
    /*
     * calculate optical flow using single level algorithm
     */
    OpticalFlowSingleLevel(img_1, img_2, kp1, kp2_singleLevel, success_singleLevel, false, false);


    /*
     * calculate optical flow using multi level algorithm
     */
    vector<cv::KeyPoint> kp2_multiLevel;
    vector<bool> success_multiLevel;
    OpticalFlowMultiLevel(img_1, img_2, kp1, kp2_multiLevel, success_multiLevel);


    cv::Mat img2_CV = opticalFlowDetectUsingOpencv(img_1, img_2, kp1);
    cv::imshow("tracked by opencv", img2_CV);

    cv::Mat img2_singleLevel = DrawOpticalFlowInImage(img_2, kp1, kp2_singleLevel, success_singleLevel);
    cv::imshow("tracked by single level", img2_singleLevel);

    cv::Mat img2_multiLevel = DrawOpticalFlowInImage(img_2, kp1, kp2_multiLevel, success_multiLevel);
    cv::imshow("tracked by multi level", img2_multiLevel);

    cv::waitKey(0);
    return 0;
}
#endif