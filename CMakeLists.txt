cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(buckling, LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set warning flags
add_compile_options(
    -O3
    -march=native
    -Wall
    -Wextra
    -Wno-unused-variable
    -Wno-unused-parameter
    -Wno-sign-compare
    -funroll-loops      # Unroll loops for potentially better performance
    -ffast-math         # Use faster but less precise math operations
    -flto               # Link-Time Optimization
    -fomit-frame-pointer # Omit frame pointer for better optimization
    -fno-omit-frame-pointer # Disable frame pointer omission if needed
    -pipe                # Use pipes rather than temporary files for compilation
)
# use fast math
add_compile_options(-ffast-math)
add_compile_options(-fopenmp-simd)


# Fetch and include pybind11
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.13.5.tar.gz
)
if(NOT TARGET pybind11)
    FetchContent_MakeAvailable(pybind11)
endif()

# Find the required packages
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

# build a shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_INCLUDE_CURRENT_DIR      ON)
SET(CMAKE_UNITY_BUILD              OFF)

# Set the source files and add module
set(SOURCE_FILES src/warpper.cpp)

# # compile the test file as test
# add_executable(test src/test.cpp)
# target_link_libraries(test PRIVATE Kokkos::kokkos)

# Add the pybind11 module
pybind11_add_module(kokkos MODULE ${SOURCE_FILES})

# Set the output directories using generator expressions
set_target_properties(kokkos PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>"
    LIBRARY_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>"
    RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>"
)

# Link the required libraries
target_link_libraries(kokkos 
  PRIVATE 
    Kokkos::kokkos
    pybind11::pybind11)

# Optimize for M1 chip if on ARM architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    add_compile_options(-mcpu=apple-m1)
endif()
