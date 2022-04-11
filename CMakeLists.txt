cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)

# optimization
set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++14 ${SSE_FLAGS} -g -O3 -march=native")


set(PYTHON_EXECUTABLE /home/lazurite/anaconda3/bin/python)
add_subdirectory(pybind11)

find_package(OpenCV 4 REQUIRED)
find_package(Sophus REQUIRED)

MESSAGE(STATUS "OpenCV version =  ${OpenCV_VERSION}")
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "Sophus path =  ${Sophus_INCLUDE_DIRS}")


include_directories(
    ${OpenCV_INCLUDE_DIRS},
    ${Sophus_INCLUDE_DIRS},
    "/usr/include/eigen3"
)

pybind11_add_module(cmake_example src/main.cpp src/imageBasic.cpp src/optical_flow.cpp src/optical_flow_bind.cpp)
target_link_libraries(cmake_example PRIVATE ${OpenCV_LIBS})


add_executable(gaussNewton src/gaussNewton.cpp)
target_link_libraries(gaussNewton PRIVATE pybind11::embed)

add_executable(opticalFlow src/optical_flow.cpp)
target_link_libraries(opticalFlow ${OpenCV_LIBS} )
# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.1
