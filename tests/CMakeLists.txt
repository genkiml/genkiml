include(${GENKIML_ROOT_DIRECTORY}/genkiml/cmake/fetch_from_git.cmake)

fetch_from_git(boost-ut https://github.com/boost-ext/ut.git)

set(executable_name genkiml_tests)

add_executable(${executable_name} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
genkiml_compile_model(${executable_name} ${CMAKE_CURRENT_SOURCE_DIR}/test_model.onnx)

target_compile_definitions(${executable_name} PRIVATE BOOST_UT_DISABLE_MODULE=1)

target_link_libraries(${executable_name}
                      PUBLIC
                      genkiml
                      ut
                      range-v3
                      )

if (MSVC)
    set(onnxruntime_dll ${CMAKE_BINARY_DIR}/_deps/onnx-src/lib/onnxruntime.dll)

    add_custom_command(TARGET ${executable_name}
            POST_BUILD
            COMMAND cmake -E copy_if_different ${onnxruntime_dll} $<TARGET_FILE_DIR:${executable_name}>)
endif()
