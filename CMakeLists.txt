cmake_minimum_required(VERSION 3.22)
project(sam_segmenter)

# Prefer a modern CUDA toolkit and build for the locally installed GPU. This must
# run before any dependency (grasp_pipeline) triggers find_package(CUDAToolkit).
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()
if(NOT CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda")
  set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root")
endif()
find_package(CUDAToolkit REQUIRED)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _sam_compute_cap
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _sam_compute_cap_status
    ERROR_QUIET)
  if(_sam_compute_cap_status EQUAL 0 AND _sam_compute_cap MATCHES "^([0-9]+)\\.([0-9]+)")
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real;121a-real")
  endif()
  message(STATUS "sam_segmenter: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Set CMake CMP0074 policy to NEW
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic> $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# Find ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_python_install_package(${PROJECT_NAME})
foreach(dep rclcpp rclcpp_components core_r1_interfaces cv_bridge sensor_msgs std_srvs std_msgs vision_msgs ament_index_cpp grasp_pipeline)
  find_package(${dep} REQUIRED)
endforeach()
add_subdirectory(third_party/sam3_cpp)
add_library(sam_segmenter_component SHARED src/sam_segmenter_node.cpp)
ament_target_dependencies(sam_segmenter_component rclcpp rclcpp_components core_r1_interfaces cv_bridge sensor_msgs std_srvs std_msgs vision_msgs ament_index_cpp grasp_pipeline)
target_link_libraries(sam_segmenter_component sam3 grasp_pipeline::model_lifecycle CUDA::cudart)
rclcpp_components_register_nodes(sam_segmenter_component "sam_segmenter::SamSegmenterNode")
add_executable(sam_segmenter_node src/main.cpp)
ament_target_dependencies(sam_segmenter_node rclcpp)
target_link_libraries(sam_segmenter_node sam_segmenter_component)
install(TARGETS sam_segmenter_component LIBRARY DESTINATION lib)
install(TARGETS sam_segmenter_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
  scripts/provision_models.sh
  scripts/sam3_segmenter_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# Install models directory if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
  install(DIRECTORY models
    DESTINATION share/${PROJECT_NAME}
  )
endif()

ament_package()
