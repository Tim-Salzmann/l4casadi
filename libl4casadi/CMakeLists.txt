cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(L4CasADi)

set(LIBTORCH_OSX_M1 https://github.com/mlverse/libtorch-mac-m1/releases/download/LibTorchOpenMP/libtorch-v2.0.0.zip)
set(LIBTORCH_OSX https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip)
set(LIBTORCH_linux https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip)

function(download_and_extract url name)
    if (EXISTS ${CMAKE_SOURCE_DIR}/${name})
        message("-- Found ${name}")
    else ()
        message("-- Downloading ${name}")
        message("${url}")
        file(DOWNLOAD ${url} ${CMAKE_SOURCE_DIR}/${name}.zip)
        message("-- Downloading ${name} - done")
        file(ARCHIVE_EXTRACT INPUT ${CMAKE_SOURCE_DIR}/${name}.zip DESTINATION ${CMAKE_SOURCE_DIR})
        file(REMOVE ${CMAKE_SOURCE_DIR}/${name}.zip)
    endif ()
endfunction()

if (APPLE)
    if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL arm64)
        message("Detected OSX M1")
        download_and_extract(${LIBTORCH_OSX_M1} libtorch)
    else()
        message("Detected OSX X86")
        download_and_extract(${LIBTORCH_OSX} libtorch)
    endif()
else()
    message("Detected Linux")
    download_and_extract(${LIBTORCH_linux} libtorch)
endif()

set(CMAKE_PREFIX_PATH libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_library(l4casadi SHARED src/l4casadi.cpp include/l4casadi.hpp)
target_include_directories(l4casadi PRIVATE include)
target_link_libraries(l4casadi ${TORCH_LIBRARIES})
set_property(TARGET l4casadi PROPERTY CXX_STANDARD 17)

install(TARGETS l4casadi LIBRARY DESTINATION l4casadi)