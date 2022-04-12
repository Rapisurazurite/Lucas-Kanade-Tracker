#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include "direct_method.h"



// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// paths
string left_file = "../res/left.png";
string disparity_file = "../res/disparity.png";
boost::format fmt_others("../res/%06d.png");    // other files


// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

#ifdef MAIN_DIRECT_METHOD
int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);
    VecVector2d pixels_ref;
    vector<double> depth_ref;
    randomSamplePoint(left_img, disparity_img, pixels_ref, depth_ref);

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        cv::Mat imShow = DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        cv::imshow((fmt_others % i).str(), imShow);
        cv::waitKey(0);
    }
    return 0;
}
#endif

void randomSamplePoint(const cv::Mat &left_img, cv::Mat &disparity_img, VecVector2d &pixels_ref,
                  vector<double> &depth_ref) {// let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;// generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }
}


cv::Mat DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);;
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;

    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }

    return img2_show;
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {

    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++) {

        // compute the projection in the second image
        Eigen::Vector3d point_ref = // p
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref; // q = T21 * p
        if (point_cur[2] < 0)   // depth invalid
            continue;

        float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;

        projection[i] = Eigen::Vector2d(u, v);
        cnt_good++;

        Matrix26d J_pixel_xi = cal_J_pixel_xi(point_cur);

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {

                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Eigen::Vector2d J_img_pixel;

                J_img_pixel = getDxDyOfImage(img2, cv::Point2f(u+x, v+y));

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    if (cnt_good) {
        // set hessian, bias and cost
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}

Matrix26d JacobianAccumulator::cal_J_pixel_xi(const Eigen::Vector3d &point_cur) const {
    double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
        Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;


    Matrix26d J_pixel_xi;
    J_pixel_xi(0, 0) = fx * Z_inv;
    J_pixel_xi(0, 1) = 0;
    J_pixel_xi(0, 2) = -fx * X * Z2_inv;
    J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
    J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
    J_pixel_xi(0, 5) = -fx * Y * Z_inv;

    J_pixel_xi(1, 0) = 0;
    J_pixel_xi(1, 1) = fy * Z_inv;
    J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
    J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
    J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
    J_pixel_xi(1, 5) = fy * X * Z_inv;
    return J_pixel_xi;
}

Eigen::Vector2d JacobianAccumulator::getDxDyOfImage(const cv::Mat &img, const cv::Point2f &p) {
    double dx, dy;
    dx = 0.5 * (GetPixelValue(img, p.x + 1, p.y) - GetPixelValue(img, p.x - 1, p.y));
    dy = 0.5 * (GetPixelValue(img, p.x, p.y + 1) - GetPixelValue(img, p.x, p.y - 1));
    return Eigen::Vector2d{dx, dy};
}

cv::Mat DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    // parameters
    int pyramids = 4;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    vector<cv::Mat> pyr1;
    vector<cv::Mat> pyr2;
    createPyramids(img1, img2, pyramids, pyr1, pyr2);

    cv::Mat matShow;
    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        matShow = DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }
    return matShow;
}

void createPyramids(const cv::Mat &img1, const cv::Mat &img2, int pyramids, vector<cv::Mat> &pyr1, vector<cv::Mat> &pyr2) {
    double pyramid_scale = 0.5;
    // create pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
}
