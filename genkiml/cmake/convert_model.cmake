function (genkiml_convert_model model_filepath output_path)
    if (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64)
        set(APPLE_M1 TRUE)
    else()
        set(APPLE_M1 FALSE)
    endif()

    find_package(Python COMPONENTS Interpreter)

    if (Python_FOUND)
        set(genkiml_root ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../..)

        # Github actions can take care of caching pip dependencies
        # Also, this virtualenv approach doesn't seem to work properly there...
        if (NOT $ENV{GITHUB_ACTIONS})
            set(venv ${CMAKE_BINARY_DIR}/.venv)

            if (NOT EXISTS ${venv})
                message("Creating virtual environment at ${venv}")
                execute_process(COMMAND ${Python_EXECUTABLE} -m venv ${venv})
            endif()

            # A trick to make sure cmake uses the virtual environment we've created
            set(ENV{VIRTUAL_ENV} ${venv})
            set(Python_FIND_VIRTUALENV FIRST)
            unset (Python_EXECUTABLE)
            find_package(Python COMPONENTS Interpreter)
            message("Python executable: ${Python_EXECUTABLE}")

            if (APPLE_M1)
                set(requirements_txt ${genkiml_root}/requirements-m1.txt)
            else ()
                set(requirements_txt ${genkiml_root}/requirements.txt)
            endif ()

            execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r ${requirements_txt})
        endif()

        if (APPLE_M1)
            set(builder_py "${Python_SITELIB}/google/protobuf/internal/builder.py")

            if (NOT EXISTS ${builder_py})
                set(builder_py_url "https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py")
                message("Getting builder.py from ${builder_py_url}")
                execute_process(COMMAND wget ${builder_py_url} -O ${builder_py})
            endif ()
        endif()

        get_filename_component(model_name ${model_filepath} NAME_WE)
        set(model_out_path ${output_path}/${model_name})
        file(MAKE_DIRECTORY ${model_out_path})
        cmake_path(CONVERT ${model_out_path} TO_NATIVE_PATH_LIST model_out_path_native)
        cmake_path(CONVERT ${model_filepath} TO_NATIVE_PATH_LIST model_path_native)

        execute_process(COMMAND ${Python_EXECUTABLE} ${genkiml_root}/genkiml.py ${model_path_native} --model-only --output-path ${model_out_path_native} COMMAND_ERROR_IS_FATAL ANY)

        file(RENAME ${model_out_path}/model.onnx ${output_path}/${model_name}.onnx)
        file(REMOVE_RECURSE ${model_out_path})

        unset(ENV{VIRTUAL_ENV} ${venv})
    else ()
        message(WARNING "Python not found")
    endif ()
endfunction ()
