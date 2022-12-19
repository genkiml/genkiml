fetch_from_git(glfw https://github.com/glfw/glfw.git)

genkiml_example(windowed_model
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                LINK_LIBRARIES range-v3 glfw
                MODEL "${CMAKE_CURRENT_SOURCE_DIR}/windowed_model.onnx")
