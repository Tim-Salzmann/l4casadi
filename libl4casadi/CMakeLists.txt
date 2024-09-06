cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(L4CasADi)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if (WIN32)
    set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set (BUILD_SHARED_LIBS TRUE)
endif ()

#set(CMAKE_PREFIX_PATH $ENV{TORCH_ENV_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_TORCH_PATH})

find_package(Torch REQUIRED)

# Load CUDA if it is installed
find_package(CUDAToolkit)
find_package(CUDA)

add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})

if (Torch_VERSION_MAJOR GREATER_EQUAL 1 AND Torch_VERSION_MINOR GREATER_EQUAL 4)
    # add_definitions(-DENABLE_TORCH_COMPILE)
endif ()
if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(l4casadi SHARED src/l4casadi.cpp include/l4casadi.hpp)

target_include_directories(l4casadi PRIVATE include)
target_include_directories(l4casadi PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(l4casadi ${TORCH_LIBRARIES})

set_property(TARGET l4casadi PROPERTY CXX_STANDARD 17)

install(TARGETS l4casadi LIBRARY DESTINATION l4casadi)