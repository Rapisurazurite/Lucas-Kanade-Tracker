//
// Created by Lazurite on 4/12/2022.
//

#ifndef CMAKE_EXAMPLE_DIRECT_METHOD_H
#define CMAKE_EXAMPLE_DIRECT_METHOD_H

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// class for accumulator jacobians in parallel
class JacobianAccumulator {
public:
    JacobianAccumulator(
            const cv::Mat &img1_,
            const cv::Mat &img2_,
            const VecVector2d &px_ref_,
            const vector<double> depth_ref_,
            Sophus::SE3d &T21_) :
            img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const { return H; }

    /// get bias
    Vector6d bias() const { return b; }

    /// get total cost
    double cost_func() const { return cost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }

    /// reset h, b, cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }
    Eigen::Vector2d getDxDyOfImage(const cv::Mat &img, const cv::Point2f &p);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;

    Matrix26d cal_J_pixel_xi(const Eigen::Vector3d &point_cur) const;
};

void randomSamplePoint(const cv::Mat &left_img,
                       cv::Mat &disparity_img,
                       VecVector2d &pixels_ref,
                       vector<double> &depth_ref);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
cv::Mat DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
cv::Mat DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
);

void createPyramids(const cv::Mat &img1, const cv::Mat &img2, int pyramids, vector<cv::Mat> &pyr1, vector<cv::Mat> &pyr2);


#endif //CMAKE_EXAMPLE_DIRECT_METHOD_H
