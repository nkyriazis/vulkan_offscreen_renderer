project(vulkan_offscreen_renderer)

cmake_minimum_required( VERSION 3.0.2 )

find_package(Vulkan)

get_filename_component(Vulkan_BIN_PATH ${Vulkan_LIBRARY} DIRECTORY)
find_program(GLSLANGVALIDATOR glslangValidator "${Vulkan_BIN_PATH}/../Bin")

macro(BuildSPIRV shader_in shader_out)
    get_filename_component(FULL_PATH_IN ${shader_in} REALPATH)
    get_filename_component(FULL_PATH_OUT ${shader_out} REALPATH)
    add_custom_command(OUTPUT ${FULL_PATH_OUT}
                       DEPENDS ${FULL_PATH_IN}
                       COMMAND ${GLSLANGVALIDATOR} -V -o ${shader_out} ${shader_in}
                       COMMENT "Building ${shader_in}"
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       VERBATIM)
endmacro()

BuildSPIRV(shaders/shader.vert shaders/shader.vert.spv)
BuildSPIRV(shaders/shader.frag shaders/shader.frag.spv)
BuildSPIRV(shaders/shader.geom shaders/shader.geom.spv)
add_custom_target(build_shaders DEPENDS 
    shaders/shader.vert.spv
    shaders/shader.frag.spv
    shaders/shader.geom.spv)

include_directories(${Vulkan_INCLUDE_DIR})
include_directories(glm)
add_executable(vulkan_example vulkan_example.cpp)
add_dependencies(vulkan_example build_shaders)
target_link_libraries(vulkan_example ${Vulkan_LIBRARIES})
set_property(TARGET vulkan_example PROPERTY CXX_STANDARD 14)

install(TARGETS vulkan_example RUNTIME DESTINATION bin)
install(FILES
    shaders/shader.vert.spv
    shaders/shader.frag.spv
    shaders/shader.geom.spv DESTINATION bin/media)