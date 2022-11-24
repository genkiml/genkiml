function (genkiml_convert_model model_filepath output_path)
    find_package(Python COMPONENTS Interpreter)

    if (Python_FOUND)
        set(venv ${CMAKE_CURRENT_BINARY_DIR}/.venv)
        message("Creating virtual environment at ${venv}")

        execute_process(COMMAND ${Python_EXECUTABLE} -m venv ${venv})
        execute_process(COMMAND source ${venv}/bin/activate)

        # TODO: Only on macOS (M1)
        set(builder_py "${Python_SITELIB}/google/protobuf/internal/builder.py")
        if (NOT EXISTS ${builder_py})
            set(builder_py_url "https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py")
            message("Getting builder.py from ${builder_py_url}")
            execute_process(COMMAND wget ${builder_py_url} -O ${builder_py})
        endif ()

        set(genkiml_root ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../)

        if (APPLE)
            set(requirements_txt ${genkiml_root}/requirements-macos.txt)
        else ()
            set(requirements_txt ${genkiml_root}/requirements.txt)
        endif ()

        cmake_path(CONVERT ${output_path} TO_NATIVE_PATH_LIST output_path_native)
        cmake_path(CONVERT ${model_filepath} TO_NATIVE_PATH_LIST model_path_native)

        execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r ${requirements_txt})
        execute_process(COMMAND ${Python_EXECUTABLE} ${genkiml_root}/genkiml.py ${model_path_native} --model-only --output-path ${output_path_native})

        get_filename_component(model_name ${model_filepath} NAME_WE)
        file(RENAME ${output_path}/model.onnx ${output_path}/${model_name}.onnx)

        execute_process(COMMAND ${venv}/bin/deactivate)
    else ()
        message(WARNING "Python not found")
    endif ()
endfunction ()
