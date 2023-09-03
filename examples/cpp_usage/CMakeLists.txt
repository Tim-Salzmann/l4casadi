cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(L4CasADiCppExecutable)

set(L4CASADI_LIB_DIR <L4CASADI_LIB_DIR>)


add_executable(l4cpp sin.cpp)

target_link_directories(l4cpp PRIVATE ${L4CASADI_LIB_DIR})
target_link_directories(l4cpp PRIVATE _l4c_generated)
target_link_libraries(l4cpp l4casadi sin_l4c)

set_property(TARGET l4cpp PROPERTY CXX_STANDARD 17)