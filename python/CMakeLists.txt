# cmake minimum required version
# Change this to your pybind11 installation path if it is not found automatically
# set(pybind11_DIR /home/ziqi/anaconda3/envs/wst/lib/python3.10/site-packages/pybind11/share/cmake/pybind11/)
# set cmake installation prefix
cmake_minimum_required(VERSION 3.20)
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /home/ziqi/anaconda3/envs/wst/lib/python3.10/site-packages/)
find_package(pybind11 CONFIG REQUIRED)
add_library(pywowlib MODULE wow_bindings.cc)
target_link_libraries(pywowlib PRIVATE pybind11::module)