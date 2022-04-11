# Lucas-Kanade Tracker implementation in C++ and call in Python

- using GaussNewtonOptimizer to optimize the Lucas-Kanade tracker
- three different implementations of the Lucas-Kanade tracker, they are:
  - Single Level Tracker
  - Multi Level Tracker with pyramid
  - OpenCV's implementation
### Build
  ```bash
  mkdir build && cd build
  cmake ..
  make all -j4
  ```
### Call in Python
- using pybind11 to wrap the C++ code 
- example in `test_module.py`