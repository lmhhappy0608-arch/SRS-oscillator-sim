cmake_minimum_required(VERSION 3.16)
project(generate_raw_grid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

find_package(ZLIB REQUIRED)
find_package(OpenMP)

add_executable(generate_raw_grid
  main.cpp
  third_party/cnpy/cnpy.cpp
)

target_include_directories(generate_raw_grid PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy
)

target_link_libraries(generate_raw_grid PRIVATE ZLIB::ZLIB)

if(OpenMP_CXX_FOUND)
  target_link_libraries(generate_raw_grid PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(generate_raw_grid PRIVATE USE_OPENMP=1)
endif()
