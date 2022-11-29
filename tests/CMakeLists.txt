include(${CMAKE_SOURCE_DIR}/genkiml/cmake/fetch_from_git.cmake)

fetch_from_git(boost-ut https://github.com/boost-ext/ut.git)

set(target genkiml_tests)

add_executable(${target} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
genkiml_compile_model(${target} ${CMAKE_CURRENT_SOURCE_DIR}/test_model.onnx)

target_compile_definitions(${target} PRIVATE BOOST_UT_DISABLE_MODULE=1)

target_link_libraries(${target}
                      PUBLIC
                      genkiml
                      ut
                      range-v3
                      )