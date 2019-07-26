cmake_minimum_required(VERSION 3.14)
project(bowler_kinematics_cppnumericalsolvers)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
if(CMAKE_COMPILER_IS_GNUCXX)
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}_coverage tests coverage)
endif()

enable_testing()

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
        ${CMAKE_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Wextra") # -fopenmp OpenMP usually makes it slower
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -Wshadow -Wnull-dereference -Og -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -Wshadow -Wnull-dereference -O3")

# Needs Eigen 3.2.10
find_package(Eigen3 3.2 REQUIRED NO_MODULE)
find_package(JNI REQUIRED)

add_executable(bowler_kinematics_cppnumericalsolvers src/solver.cpp src/main.cpp)

target_include_directories(bowler_kinematics_cppnumericalsolvers PUBLIC include)
target_include_directories(bowler_kinematics_cppnumericalsolvers PUBLIC ${JNI_INCLUDE_DIRS})
target_link_libraries(bowler_kinematics_cppnumericalsolvers Eigen3::Eigen gtest_main)
