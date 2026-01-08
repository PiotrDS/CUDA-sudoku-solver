cmake_minimum_required(VERSION 3.20)
project(kmeans_cuda LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()




add_executable(sudoku_solver
    src/host/main.cpp
    src/device/solver_kernel.cu
    src/host/utils.cpp
)

target_compile_options(sudoku_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)


set_target_properties(sudoku_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON )


target_include_directories(sudoku_solver PRIVATE
    ${PROJECT_SOURCE_DIR}/src/device
    ${PROJECT_SOURCE_DIR}/src/include
)
target_link_libraries(sudoku_solver PRIVATE ${CUDA_LIBRARIES})