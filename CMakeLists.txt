project(vulkan_offscreen_renderer)

cmake_minimum_required( VERSION 3.0.2 )

find_package(Vulkan)

include_directories(${Vulkan_INCLUDE_DIR})
add_executable(vulkan_example vulkan_example.cpp)
target_link_libraries(vulkan_example ${Vulkan_LIBRARIES})
set_property(TARGET vulkan_example PROPERTY CXX_STANDARD 14)