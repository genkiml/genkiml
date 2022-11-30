function (genkiml_convert_model model_filepath output_path)
    set(genkiml_root ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../..)

    get_filename_component(model_name ${model_filepath} NAME_WE)
    set(model_out_path ${output_path}/${model_name})
    file(MAKE_DIRECTORY ${model_out_path})
    cmake_path(CONVERT ${model_out_path} TO_NATIVE_PATH_LIST model_out_path_native)
    cmake_path(CONVERT ${model_filepath} TO_NATIVE_PATH_LIST model_path_native)

    execute_process(COMMAND ${Python_EXECUTABLE} ${genkiml_root}/genkiml.py ${model_path_native} --model-only --output-path ${model_out_path_native} COMMAND_ERROR_IS_FATAL ANY)

    file(RENAME ${model_out_path}/model.onnx ${output_path}/${model_name}.onnx)
    file(REMOVE_RECURSE ${model_out_path})
endfunction ()
