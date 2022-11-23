function(genki_ml_example example_name model_filepath)
    genki_ml_convert_model(${model_filepath})

    set(executable_name genki_ml_${example_name})

    add_executable(${executable_name} ${CMAKE_CURRENT_SOURCE_DIR}/${example_name}.cpp)
    target_link_libraries(${executable_name} genki_ml)

    # TODO: Figure out how to do this less explicitly
    if (MSVC)
        set(onnxruntime_dll ${CMAKE_BINARY_DIR}/_deps/onnx-src/lib/onnxruntime.dll)

        add_custom_command(TARGET ${executable_name}
                POST_BUILD
                COMMAND cmake -E copy_if_different ${onnxruntime_dll} $<TARGET_FILE_DIR:${executable_name}>)
    endif()
endfunction()

genki_ml_example(single_sample "${CMAKE_CURRENT_SOURCE_DIR}/model_tf.onnx")
