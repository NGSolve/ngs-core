
cmake_minimum_required(VERSION 3.0)

project(ngs_core)

set(NGS_CORE_CPP_FILES exception.cpp localheap.cpp paje_interface.cpp profiler.cpp
  table.cpp taskmanager.cpp bitarray.cpp)

set(CMAKE_CXX_STANDARD 14)

add_library(ngs_core ${NGS_CORE_CPP_FILES})
