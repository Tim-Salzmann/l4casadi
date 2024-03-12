cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(L4CasADi)

# Load CUDA if it is installed
find_package(CUDAToolkit)
find_package(CUDA)

if (WIN32)
    set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set (BUILD_SHARED_LIBS TRUE)
endif ()

#set(CMAKE_PREFIX_PATH $ENV{TORCH_ENV_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_TORCH_PATH})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(l4casadi SHARED src/l4casadi.cpp include/l4casadi.hpp)

target_include_directories(l4casadi PRIVATE include)
target_include_directories(l4casadi PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(l4casadi ${TORCH_LIBRARIES})

set_property(TARGET l4casadi PROPERTY CXX_STANDARD 17)

install(TARGETS l4casadi LIBRARY DESTINATION l4casadi)