cmake_minimum_required(VERSION 3.22)
project(Example_class VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER "nvc++")

set(NAME "task")

message(STATUS "Compile C++:" ${CMAKE_CXX_COMPILER})

find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(${NAME} "task.cu")

target_compile_features(${NAME} PRIVATE cxx_std_20)

target_compile_options(${NAME} PRIVATE -arch=native)

target_link_libraries(${NAME} PRIVATE Boost::program_options)

target_include_directories(${NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

message(STATUS "Configuration completed")