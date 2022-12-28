project(yylvv_2_0_tools)

find_package(glm REQUIRED)

add_executable(yylvv_2_0_counter counter.cpp)
add_executable(yylvv_2_0_drop_first drop_first.cpp)
add_executable(yylvv_2_0_cubify cubify.cpp)

target_link_libraries(yylvv_2_0_cubify glm::glm)
