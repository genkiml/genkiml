cmake_minimum_required(VERSION 3.17)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")
set(CMAKE_CXX_STANDARD 17)

project(GenkiML VERSION 0.1.0)

message("CMake system name: ${CMAKE_SYSTEM_NAME}, is apple? ${APPLE}")

option(GENKIML_BUILD_TESTS "Build Genki ML examples" ON)
option(GENKIML_BUILD_EXAMPLES "Build Genki ML examples" ON)
option(GENKIML_BUILD_ONNXRUNTIME_STATIC_LIB "Build and link static library for onnxruntime" OFF)

find_package(Python COMPONENTS Interpreter)

if (Python_FOUND)
    execute_process(COMMAND ${Python_EXECUTABLE} --version)
endif()

#add_subdirectory(genkiml)
#
#if (${GENKIML_BUILD_EXAMPLES})
#    add_subdirectory(examples)
#endif ()
#
#if (GENKIML_BUILD_TESTS)
#    add_subdirectory(tests)
#endif()
