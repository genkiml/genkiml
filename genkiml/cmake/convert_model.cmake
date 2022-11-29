function (genkiml_convert_model model_filepath output_path)
    find_package(Python COMPONENTS Interpreter)

    if (Python_FOUND)
        set(venv ${CMAKE_CURRENT_BINARY_DIR}/.venv)
        message("Creating virtual environment at ${venv}")

        execute_process(COMMAND ${Python_EXECUTABLE} -m venv ${venv})

        if (MSVC)
            cmake_path(CONVERT ${venv}/Scripts/activate.bat TO_NATIVE_PATH_LIST venv_activate_bat)
            execute_process(COMMAND ${venv_activate_bat})
        else ()
            execute_process(COMMAND source ${venv}/bin/activate)
        endif ()

        set(genkiml_root ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../)
        set(APPLE_M1 ${APPLE} AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64)

        message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
        message("CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
        message("APPLE: ${APPLE}")
        message("APPLE_M1: ${APPLE_M1}")

        if (APPLE_M1)
            set(requirements_txt ${genkiml_root}/requirements-m1.txt)
        else ()
            set(requirements_txt ${genkiml_root}/requirements.txt)
        endif ()

        get_filename_component(model_name ${model_filepath} NAME_WE)

        set(model_out_path ${output_path}/${model_name})
        file(MAKE_DIRECTORY ${model_out_path})
        cmake_path(CONVERT ${model_out_path} TO_NATIVE_PATH_LIST model_out_path_native)
        cmake_path(CONVERT ${model_filepath} TO_NATIVE_PATH_LIST model_path_native)

        execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r ${requirements_txt})

        if (APPLE_M1)
            set(builder_py "${Python_SITELIB}/google/protobuf/internal/builder.py")

            if (NOT EXISTS ${builder_py})
                set(builder_py_url "https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py")
                message("Getting builder.py from ${builder_py_url}")
                execute_process(COMMAND wget ${builder_py_url} -O ${builder_py})
            endif ()
        endif()

        execute_process(COMMAND ${Python_EXECUTABLE} ${genkiml_root}/genkiml.py ${model_path_native} --model-only --output-path ${model_out_path_native})

        file(RENAME ${model_out_path}/model.onnx ${output_path}/${model_name}.onnx)
        file(REMOVE_RECURSE ${model_out_path})

        execute_process(COMMAND ${venv}/bin/deactivate)
    else ()
        message(WARNING "Python not found")
    endif ()
endfunction ()
