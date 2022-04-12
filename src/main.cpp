//
// Created by Lazurite on 4/9/2022.
//

#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
void init_ex_imageBasic(py::module_ &m);
void init_ex_opticalFlow(py::module &m);
void init_ex_directMethod(py::module &m);

PYBIND11_MODULE(cmake_example, m) {
m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";


/*
 * FUNCTION REGISTRATION
 */
init_ex_imageBasic(m);
init_ex_opticalFlow(m);
init_ex_directMethod(m);




#ifdef VERSION_INFO
m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
m.attr("__version__") = "dev";
#endif
}
