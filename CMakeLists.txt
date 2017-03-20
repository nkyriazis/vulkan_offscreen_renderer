project(vulkan_offscreen_renderer)

cmake_minimum_required( VERSION 3.0.2 )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

find_package(Vulkan)
find_package(Shaderc)

include_directories(${Vulkan_INCLUDE_DIR})
include_directories(${SHADERC_INCLUDE_DIR})
include_directories(glm)
add_executable(vulkan_example vulkan_example.cpp)
add_dependencies(vulkan_example build_shaders)
target_link_libraries(vulkan_example ${Vulkan_LIBRARIES} ${SHADERC_LIBRARIES})
set_property(TARGET vulkan_example PROPERTY CXX_STANDARD 14)
install(TARGETS vulkan_example RUNTIME DESTINATION bin)
