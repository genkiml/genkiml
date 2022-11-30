find_package(Python COMPONENTS Interpreter REQUIRED)
message("Default python interpreter: ${Python_EXECUTABLE}")

if (NOT DEFINED ENV{PYTHON_VENV})
    set(venv ${CMAKE_SOURCE_DIR}/.venv)
else()
    set(venv $ENV{PYTHON_VENV})
endif()

if (NOT EXISTS ${venv})
    message("Creating virtual environment at ${venv}")
    execute_process(COMMAND ${Python_EXECUTABLE} -m venv ${venv} COMMAND_ERROR_IS_FATAL ANY)
endif()

message("Using Python virtual environment: ${venv}")
unset (Python_EXECUTABLE)

if (MSVC)
    set(Python_ROOT_DIR ${venv}/Scripts)
else()
    set(Python_ROOT_DIR ${venv}/bin)
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)

if (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64)
    set(APPLE_M1 TRUE)
else()
    set(APPLE_M1 FALSE)
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)
message("Using Python interpreter: ${Python_EXECUTABLE}")

if (APPLE_M1)
    set(requirements_txt ${CMAKE_SOURCE_DIR}/requirements-m1.txt)
else ()
    set(requirements_txt ${CMAKE_SOURCE_DIR}/requirements.txt)
endif ()

execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r ${requirements_txt})

if (APPLE_M1)
    set(builder_py "${Python_SITELIB}/google/protobuf/internal/builder.py")

    if (NOT EXISTS ${builder_py})
        set(builder_py_url "https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py")
        message("Getting builder.py from ${builder_py_url}")
        execute_process(COMMAND wget ${builder_py_url} -O ${builder_py})
    endif ()
endif()

