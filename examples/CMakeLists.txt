function(genkiml_example example_name)
    set(flag_args "")
    set(single_value_args MODEL)
    set(multi_value_args SOURCES LINK_LIBRARIES)
    cmake_parse_arguments(PARSED_ARGS "${flag_args}" "${single_value_args}" "${multi_value_args}" ${ARGN})

    set(executable_name genkiml_example_${example_name})

    add_executable(${executable_name} ${PARSED_ARGS_SOURCES})
    genkiml_compile_model(${executable_name} ${PARSED_ARGS_MODEL})
    target_link_libraries(${executable_name} PUBLIC genkiml ${PARSED_ARGS_LINK_LIBRARIES})

    # TODO: Figure out how to do this less explicitly
    if (MSVC)
        set(onnxruntime_dll ${CMAKE_BINARY_DIR}/_deps/onnx-src/lib/onnxruntime.dll)

        add_custom_command(TARGET ${executable_name}
                POST_BUILD
                COMMAND cmake -E copy_if_different ${onnxruntime_dll} $<TARGET_FILE_DIR:${executable_name}>)
    endif()
endfunction()

add_subdirectory(onnx)
add_subdirectory(keras)
add_subdirectory(windowed_model)