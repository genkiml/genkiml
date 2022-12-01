cmake_minimum_required(VERSION 3.17)

set(genkiml_lib genkiml)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_from_git.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/convert_model.cmake)

set(ONNXRUNTIME_VERSION "1.13.1")

if (NOT TARGET fmt)
    fetch_from_git(fmt "https://github.com/fmtlib/fmt.git")
endif ()

if (NOT TARGET range-v3)
    fetch_from_git(range-v3 "https://github.com/ericniebler/range-v3.git")
endif ()

if (NOT TARGET GSL)
    fetch_from_git(gsl "https://github.com/microsoft/GSL.git" GIT_TAG main)
endif ()

fetch_from_git(cmrc https://github.com/vector-of-bool/cmrc.git ADD_SUBDIRECTORY)

if (NOT GENKIML_BUILD_ONNXRUNTIME_STATIC_LIB)
    message("Using dynamic library for onnxruntime")

    include(cmake/fetch_onnxruntime_dynamic_lib.cmake)
    fetch_onnxruntime_dynamic_lib(${ONNXRUNTIME_VERSION})

    set(onnxruntime_libs onnxruntime)
    add_library(onnxruntime_libs INTERFACE)
    target_link_libraries(onnxruntime_libs
                          INTERFACE
                          onnxruntime
                          GSL
                          )
else ()
    message("Building static library for onnxruntime")

    fetch_from_git(onnxruntime "https://github.com/microsoft/onnxruntime.git"
                   GIT_TAG main
                   SOURCE_DIRECTORY_VAR onnxruntime_source_dir
                   BINARY_DIRECTORY_VAR onnxruntime_binary_dir)

    set(onnxruntime_USE_CUDA OFF)
    set(onnxruntime_BUILD_UNIT_TESTS OFF)
    add_subdirectory(${onnxruntime_source_dir}/cmake ${onnxruntime_binary_dir})

    set(onnxruntime_libs onnxruntime)
    add_library(onnxruntime_libs INTERFACE)

    target_include_directories(onnxruntime_libs
                               INTERFACE
                               ${onnxruntime_source_dir}/include/onnxruntime/core/session
                               )

    # TODO: Look into the minimal set of libraries to link for our use of onnxruntime
    target_link_libraries(onnxruntime_libs
                          INTERFACE
                          onnxruntime_common
                          onnxruntime_flatbuffers
                          onnxruntime_framework
                          onnxruntime_graph
                          onnxruntime_mlas
                          onnxruntime_optimizer
                          onnxruntime_providers
                          onnxruntime_session
                          onnxruntime_util
                          re2
                          absl::hash
                          absl::container_common
                          absl::raw_hash_set
                          onnx
                          onnx_proto
                          nsync
                          nsync_cpp
                          )
endif ()

add_library(${genkiml_lib}
            STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/source/genkiml.cpp)

target_include_directories(${genkiml_lib}
                           PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}/source
                           )

target_link_libraries(${genkiml_lib}
                      PUBLIC
                      fmt
                      GSL
                      range-v3
                      onnxruntime_libs
                      )

function (genkiml_compile_model target_name model_filepath)
    set(model_out_dir ${CMAKE_BINARY_DIR})
    genkiml_convert_model(${model_filepath} ${model_out_dir})
    get_filename_component(model_name ${model_filepath} NAME_WE)

    set(model_lib _${model_name})
    set(genkiml_src ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../genkiml/source)

    cmrc_add_resource_library(${model_lib}
                              NAMESPACE files
                              WHENCE ${model_out_dir}
                              PREFIX models
                              ${model_out_dir}/${model_name}.onnx
                              )

    target_sources(${model_lib} PRIVATE ${genkiml_src}/load_model.cpp)
    target_link_libraries(${model_lib} PUBLIC fmt GSL)

    target_link_libraries(${target_name} PUBLIC ${model_lib})
endfunction ()
