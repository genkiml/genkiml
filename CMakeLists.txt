cmake_minimum_required(VERSION 3.17)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")
set(CMAKE_CXX_STANDARD 17)

project(GenkiML VERSION 0.1.0)

option(GENKIML_BUILD_TESTS "Build Genki ML examples" ON)
option(GENKIML_BUILD_EXAMPLES "Build Genki ML examples" ON)
option(GENKIML_BUILD_ONNXRUNTIME_STATIC_LIB "Build and link static library for onnxruntime" OFF)

#=======================================================================================================================
find_package(Python COMPONENTS Interpreter REQUIRED)
message("Default python interpreter: ${Python_EXECUTABLE}")

if (NOT DEFINED ENV{PYTHON_VENV})
    set(venv ${CMAKE_SOURCE_DIR}/.venv)

    if (NOT EXISTS ${venv})
        message("Creating virtual environment at ${venv}")
        execute_process(COMMAND ${Python_EXECUTABLE} -m venv ${venv} COMMAND_ERROR_IS_FATAL ANY)
    endif()
else()
    set(venv $ENV{PYTHON_VENV})
endif()

message("Using Python virtual environment: ${venv}")
unset (Python_EXECUTABLE)

if (MSVC)
    set(Python_ROOT_DIR ${venv}/Scripts)
else()
    set(Python_ROOT_DIR ${venv}/bin)
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)

#=======================================================================================================================
add_subdirectory(genkiml)

if (${GENKIML_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif ()

if (${GENKIML_BUILD_TESTS})
    add_subdirectory(tests)
endif()
