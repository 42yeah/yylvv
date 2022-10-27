project(yylvv_nrrd)
find_package(glm REQUIRED)
add_library(yylvv_nrrd NRRD.cpp NRRD.h VectorField.cpp VectorField.h)
target_link_libraries(yylvv_nrrd PUBLIC
        glm::glm)
# maybe my macOS's glm cmake is broken
if (APPLE)
    target_include_directories(yylvv_nrrd PUBLIC
            /usr/local/Cellar/glm/0.9.9.8/include)
    set(YYLVV_GLM_INC /usr/local/Cellar/glm/0.9.9.8/include)
else()
    target_include_directories(yylvv_nrrd PUBLIC
            glm::glm)
    set(YYLVV_GLM_INC "")
endif()
set_target_properties(yylvv_nrrd PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR};${YYLVV_GLM_INC}"
        INTERFACE_LINK_LIBRARIES "glm::glm")
