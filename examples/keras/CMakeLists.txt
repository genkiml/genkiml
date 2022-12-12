execute_process(COMMAND ${Python_EXECUTABLE} generate_model.py --model-only
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMAND_ERROR_IS_FATAL ANY
                )

genkiml_example(keras
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                MODEL "${CMAKE_CURRENT_SOURCE_DIR}/fc_model")
