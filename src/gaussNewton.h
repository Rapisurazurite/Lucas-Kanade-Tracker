//
// Created by Lazurite on 4/9/2022.
//

#ifndef CMAKE_EXAMPLE_GAUSSNEWTON_H
#define CMAKE_EXAMPLE_GAUSSNEWTON_H

namespace py = pybind11;

class gaussNewton {
public:
    double ar, br, cr;
    double ae, be, ce;
    int N;

    gaussNewton(double ar, double br, double cr, double ae, double be, double ce) : ar(ar), br(br), cr(cr), ae(ae),
                                                                                    be(be), ce(ce), N(100) {};

    ~gaussNewton() = default;

    double f(double x) {
        return exp(ar * x * x + br * x + cr);
    }

    void initRealData(std::vector<double> &x_data, std::vector<double> &y_data);

    Eigen::Vector3d getDerivative(double &cost, std::vector<double> &x_data, std::vector<double> &y_data);

    std::vector<double> solve();
};

extern void init_ex_gaussNewton(py::module &m);

#endif //CMAKE_EXAMPLE_GAUSSNEWTON_H
