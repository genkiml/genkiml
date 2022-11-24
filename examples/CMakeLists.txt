function(genkiml_example example_name model_filepath)
    set(executable_name genkiml_${example_name})

    add_executable(${executable_name} ${CMAKE_CURRENT_SOURCE_DIR}/${example_name}/main.cpp)
    genkiml_link_model(${executable_name} ${model_filepath})

    # TODO: Figure out how to do this less explicitly
    if (MSVC)
        set(onnxruntime_dll ${CMAKE_BINARY_DIR}/_deps/onnx-src/lib/onnxruntime.dll)

        add_custom_command(TARGET ${executable_name}
                POST_BUILD
                COMMAND cmake -E copy_if_different ${onnxruntime_dll} $<TARGET_FILE_DIR:${executable_name}>)
    endif()
endfunction()

genkiml_example(keras_model "${CMAKE_CURRENT_SOURCE_DIR}/keras_model/fc_model")
