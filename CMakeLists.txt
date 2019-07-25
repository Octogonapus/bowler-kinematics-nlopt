cmake_minimum_required(VERSION 3.14)
project(bowler_kinematics_cppnumericalsolvers)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra") # -fopenmp OpenMP usually makes it slower
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Needs Eigen 3.2.10
find_package(Eigen3 3.2 REQUIRED NO_MODULE)
find_package(JNI REQUIRED)

add_executable(bowler_kinematics_cppnumericalsolvers main.cpp)

target_include_directories(bowler_kinematics_cppnumericalsolvers PUBLIC include)
target_include_directories(bowler_kinematics_cppnumericalsolvers PUBLIC ${JNI_INCLUDE_DIRS})
target_link_libraries(bowler_kinematics_cppnumericalsolvers Eigen3::Eigen)
