cmake_minimum_required(VERSION 3.8)

project(sad_cpu_gpu LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)
add_library(${PROJECT_NAME} STATIC
    src/disparity.cpp
    src/disparity.cu
)
# Request that the library be built with -std=c++11
# As this is a public compile feature anything that links to 
# the library will also build with -std=c++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the 
# library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)
                    
add_executable(disparity main.cpp)
target_link_libraries(disparity ${PROJECT_NAME})    