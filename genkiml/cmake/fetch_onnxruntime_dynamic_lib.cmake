function (fetch_onnxruntime_dynamic_lib onnx_lib_version)

    if (APPLE OR MSVC OR UNIX)
        set(base_url "https://github.com/microsoft/onnxruntime/releases/download/v${onnx_lib_version}")

        if (APPLE)
            set(onnx_url "${base_url}/onnxruntime-osx-universal2-${onnx_lib_version}.tgz")
        elseif (MSVC)
            set(onnx_url "${base_url}/onnxruntime-win-x64-${onnx_lib_version}.zip")
        elseif (UNIX)
            if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL x86_64)
                set(arch x64)
            elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL aarch64)
                set(arch aarch64)
            else()
                message(FATAL_ERROR "Unsupported platform for onnxruntime shared library")
            endif()

            set(onnx_url "${base_url}/onnxruntime-linux-${arch}-${onnx_lib_version}.tgz")
        endif ()

        message("Fetching onnxruntime from ${onnx_url}")

        include(FetchContent)

        FetchContent_Declare(onnx URL ${onnx_url})
        FetchContent_MakeAvailable(onnx)
        FetchContent_GetProperties(onnx)

        if (NOT onnx_POPULATED)
            FetchContent_Populate(onnx)
        endif ()

        message("ONNX source: ${onnx_SOURCE_DIR}")

        add_library(onnxruntime INTERFACE)
        target_include_directories(onnxruntime INTERFACE ${onnx_SOURCE_DIR}/include)

        if (APPLE)
            target_link_libraries(onnxruntime INTERFACE ${onnx_SOURCE_DIR}/lib/libonnxruntime.${onnx_lib_version}.dylib)
        elseif (MSVC)
            target_link_libraries(onnxruntime INTERFACE ${onnx_SOURCE_DIR}/lib/onnxruntime.lib)
        elseif (UNIX)
            target_link_libraries(onnxruntime INTERFACE ${onnx_SOURCE_DIR}/lib/libonnxruntime.so.${onnx_lib_version})
        endif ()
    endif ()

endfunction ()