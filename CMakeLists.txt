cmake_minimum_required(VERSION 3.17)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")
set(CMAKE_CXX_STANDARD 17)

project(GenkiML VERSION 0.1.0)

option(GENKIML_BUILD_TESTS "Build Genki ML examples" ON)
option(GENKIML_BUILD_EXAMPLES "Build Genki ML examples" ON)
option(GENKIML_BUILD_ONNXRUNTIME_STATIC_LIB "Build and link static library for onnxruntime" OFF)

set(GENKIML_ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
include(${CMAKE_CURRENT_SOURCE_DIR}/genkiml/cmake/genkiml_python.cmake)

add_subdirectory(genkiml)

if (${GENKIML_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif ()

if (${GENKIML_BUILD_TESTS})
    add_subdirectory(tests)
endif()
