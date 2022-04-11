//
// Created by Lazurite on 4/9/2022.
//
#include <iostream>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "gaussNewton.h"

using namespace std;


void gaussNewton::initRealData(vector<double> &x_data, vector<double> &y_data) {
    x_data.clear();
    y_data.clear();
    for (int i = 0; i < N; i++) {
        double x = i * 1.0 / N;
        x_data.push_back(x);
        y_data.push_back(f(x));
    }
}

Eigen::Vector3d gaussNewton::getDerivative(double &cost, vector<double> &x_data, vector<double> &y_data) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    cost = 0.0;

    for (int i = 0; i < N; i++) {
        double xi = x_data[i];
        double yi = y_data[i];
        double error = yi - exp(ae * xi * xi + be * xi + ce);
        Eigen::Vector3d J; //残差函数的导数
        J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce); //d(y-y_hat)/d(a)
        J[1] = -xi * exp(ae * xi * xi + be * xi + ce); //d(y-y_hat)/d(b)
        J[2] = -exp(ae * xi * xi + be * xi + ce); //d(y-y_hat)/d(c)

        H += J * J.transpose(); //Hessian矩阵
        g += -error * J; //偏差
        cost += error * error; //代价函数值
    }
    Eigen::Vector3d dx = H.ldlt().solve(g); //求解线性方程
    return dx;
}

std::vector<double> gaussNewton::solve() {
    vector<double> x_data(N), y_data(N);
    initRealData(x_data, y_data);

    int iterations = 100;
    double cost = 0, lastCost = 0;

    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        Eigen::Vector3d dx = getDerivative(cost, x_data, y_data);
        if (isnan(dx[0])) {
            py::print("result is nan");
            break;
        }
        if (iter > 0 && cost >= lastCost) {
            py::print("cost is increasing");
            py::print("iteration: ", iter, " cost: ", cost, " lastCost: ", lastCost);
            break;
        }
        //更新估计参数
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;
        py::print("iteration: ", iter, "cost: ", cost);
        py::print("ae: ", ae, "be: ", be, "ce: ", ce);
    }
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(end - start);
    py::print("time used: ", time_used.count(), "s");
    py::print("last cost: ", lastCost);
    py::print("ae: ", ae, " be: ", be, " ce: ", ce);

    vector<double> result;
    result.push_back(ae);
    result.push_back(be);
    result.push_back(ce);
    return result;
}


void init_ex_gaussNewton(py::module &m) {
    py::class_<gaussNewton>(m, "gaussNewton")
            .def(py::init<double, double, double, double, double, double>(),
                 py::arg("ar") = 1.0, py::arg("br") = 2.0, py::arg("cr") = 1.0,
                 py::arg("ae") = 2.0, py::arg("be") = -4.0, py::arg("ce") = 5.0)
             .def("solve", &gaussNewton::solve, "fit the parameters");
}

int main() {
    py::scoped_interpreter guard{};
    auto g = gaussNewton(1.0, 2.0, 1.0, 2.0, -4.0, 5.0);
    g.solve();
}