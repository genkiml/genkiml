cmake_minimum_required(VERSION 3.17)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")
set(CMAKE_CXX_STANDARD 17)

project(GenkiML VERSION 0.1.0)

option(GENKI_ML_BUILD_EXAMPLES "Build Genki ML examples" OFF)
option(GENKI_ML_BUILD_ONNXRUNTIME_STATIC_LIB "Build and link static library for onnxruntime" OFF)

add_subdirectory(genkiml)

if (${GENKI_ML_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif ()
