cmake_minimum_required(VERSION 3.19.0)
project(yylvv_2_0 CUDA CXX)

# source files
set(yylvv_2_0_SRC main.cu glad/src/glad.c VAO.cpp Program.cpp Camera.cpp renderstates/lineglyph.cu renderstates/arrowglyph.cu debug_kernels.cu renderstates/streamline.cu app.cu)
set_source_files_properties(${yylvv_2_0_SRC} PROPERTIES LANGUAGE CUDA)

set(CMAKE_CXX_STANDARD 17)

# locate GLFW and Co.
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_executable(yylvv_2_0 ${yylvv_2_0_SRC})
target_compile_options(yylvv_2_0 PUBLIC
    -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)

# NRRD dependency
add_subdirectory(nrrd)

target_link_libraries(yylvv_2_0 PUBLIC
    glfw
    glm::glm
    yylvv_nrrd)
target_include_directories(yylvv_2_0 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/glad/include)
