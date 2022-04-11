//
// Created by Lazurite on 4/11/2022.
//

#ifndef CMAKE_EXAMPLE_OPTICAL_FLOW_H
#define CMAKE_EXAMPLE_OPTICAL_FLOW_H

class OpticalFlowTracker {
private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const std::vector<cv::KeyPoint> &kp1;
    std::vector<cv::KeyPoint> &kp2;
    std::vector<bool> &success;
    bool inverse;
    bool has_initial;
public:
    OpticalFlowTracker(
            const cv::Mat &img_1,
            const cv::Mat &img_2,
            const std::vector<cv::KeyPoint> &kp1,
            std::vector<cv::KeyPoint> &kp2,
            std::vector<bool> &success,
            bool inverse,
            bool has_initial
    ) : img1(img_1), img2(img_2), kp1(kp1), kp2(kp2), success(success), inverse(inverse), has_initial(has_initial) {};

    virtual ~OpticalFlowTracker() = default;

    void calculateOpticalFlow(const cv::Range &range);

    Eigen::Vector2d getDxDyOfImage(const cv::Mat &img, const cv::Point2f &p);

    Eigen::Vector3d getDerivative(double &cost);

    double calBatchUpdate(const int half_patch_size, const cv::KeyPoint &kp, double dx, double dy,
                          Eigen::Vector2d &J, Eigen::Matrix2d &H, Eigen::Vector2d &b,
                          Eigen::Vector2d &update, int iter);
};

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 */
void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<cv::KeyPoint> &kp1,
        std::vector<cv::KeyPoint> &kp2,
        std::vector<bool> &success,
        bool inverse = false,
        bool has_initial = false
);

void OpticalFlowMultiLevel(
        const cv::Mat &img_1,
        const cv::Mat &img_2,
        const std::vector<cv::KeyPoint> &kp1,
        std::vector<cv::KeyPoint> &kp2_multiLevel,
        std::vector<bool> &success_multiLevel);

template<typename PointType, typename successType>
cv::Mat DrawOpticalFlowInImage(cv::Mat &img, std::vector<PointType> &kp1, std::vector<PointType> &kp2, std::vector<successType> &success) {
    assert(kp1.size() == kp2.size());
    cv::Mat ImageRBG;
    cv::cvtColor(img, ImageRBG, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < kp1.size(); i++) {
        if (success[i]) {
            cv::circle(ImageRBG, kp2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(ImageRBG, kp1[i], kp2[i], cv::Scalar(0, 250, 0));
        }
    }
    return ImageRBG;
}

cv::Mat DrawOpticalFlowInImage(cv::Mat &img, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2, std::vector<bool> &success);

/**
 * using opencv to calculate optical flow
 * @param img_1
 * @param img_2
 * @param kp1 the keypoints in img1
 * @return
 */
cv::Mat opticalFlowDetectUsingOpencv(cv::Mat &img_1, cv::Mat &img_2, std::vector<cv::KeyPoint> &kp1);

#endif //CMAKE_EXAMPLE_OPTICAL_FLOW_H
