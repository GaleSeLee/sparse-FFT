cmake_minimum_required(VERSION 3.18)

project(sparseFFT)

set(PROJECT_BIN_NAME sparseFFT)

include_directories(./include)

add_executable(${PROJECT_BIN_NAME}
                source/main.cpp
                source/common.cpp
                source/baseline.cpp
                source/accelerate.cpp
                )

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -g -ggdb")

include(CheckLanguage)
check_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
enable_language(CUDA)
set_property(
    TARGET ${ABACUS_BIN_NAME}
    PROPERTY CUDA_ARCHITECTURES
    70 # V100
    80 # A100
)
target_link_libraries(
    ${PROJECT_BIN_NAME}
    -lcufft
    -lcudart
    -lnvToolsExt
)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
