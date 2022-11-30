function(genkiml_example example_name model_filepath)
    set(executable_name genkiml_example_${example_name})

    add_executable(${executable_name} ${CMAKE_CURRENT_SOURCE_DIR}/${example_name}/main.cpp)
    genkiml_compile_model(${executable_name} ${model_filepath})
    target_link_libraries(${executable_name} PUBLIC genkiml)

    # TODO: Figure out how to do this less explicitly
    if (MSVC)
        set(onnxruntime_dll ${CMAKE_BINARY_DIR}/_deps/onnx-src/lib/onnxruntime.dll)

        add_custom_command(TARGET ${executable_name}
                POST_BUILD
                COMMAND cmake -E copy_if_different ${onnxruntime_dll} $<TARGET_FILE_DIR:${executable_name}>)
    endif()
endfunction()

genkiml_example(onnx "${CMAKE_CURRENT_SOURCE_DIR}/onnx/model.onnx")

execute_process(COMMAND ${Python_EXECUTABLE} generate_model.py --model-only
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/keras
                COMMAND_ERROR_IS_FATAL ANY
                )

genkiml_example(keras "${CMAKE_CURRENT_SOURCE_DIR}/keras/fc_model")
