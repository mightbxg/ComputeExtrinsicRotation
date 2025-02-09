cmake_minimum_required(VERSION 3.10)

project(ComputeExtrinsicRotation)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

add_executable(t t.cpp)
target_link_libraries(t Eigen3::Eigen)
