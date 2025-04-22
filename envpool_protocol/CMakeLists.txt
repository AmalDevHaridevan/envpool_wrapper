#    Copyright 2025 Amaldev Haridevan

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(envpool_protocol LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

set(ENVPOOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/envpool_deps" CACHE STRING "Directory to envpool")
set(THREADPOOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/envpool_deps" CACHE STRING "Directory to threadpool")
set(CONCURRENT_QUEUE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/envpool_deps" CACHE STRING "Directory to concurrent queue")
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(CUDAToolkit REQUIRED)
link_directories(${CUDAToolkit_LIBRARY_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${THREADPOOL_DIR})
include_directories(${CONCURRENT_QUEUE_DIR})
include_directories(${ENVPOOL_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${EIGEN3_INCLUDE_DIRS})
# obtain the module name
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -E -dM ${CMAKE_CURRENT_SOURCE_DIR}/include/definitions.hh
    OUTPUT_VARIABLE PREPROCESSED_CODE
)
string(REPLACE "\n" ";" PREPROCESSED_LINES "${PREPROCESSED_CODE}")
foreach(line IN LISTS PREPROCESSED_LINES)
    string(REGEX MATCH "^#define[ \t]+MODULE_NAME[ \t]+([A-Za-z0-9_]+)[ \t]*$" MATCHED "${line}")
    if(MATCHED)
        string(REGEX REPLACE "^#define[ \t]+MODULE_NAME[ \t]+([A-Za-z0-9_]+)[ \t]*$" "\\1" MODULE_NAME "${line}")
        break()
    endif()
endforeach()
if (MODULE_NAME STREQUAL "")
    message(FATAL_ERROR "Module name not found in definitions.hh")
endif()
message(STATUS "Module name: " ${MODULE_NAME})
# string(REGEX MATCH "^#define( |\t)+CLSNAME( |\t)+[A-Za-z0-9_]+( |\t)*$" MODULE_NAME PREPROCESSED_CODE)

pybind11_add_module(${MODULE_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper.cc
    )
set_target_properties(${MODULE_NAME} PROPERTIES
OUTPUT_NAME ${MODULE_NAME}
)
target_link_libraries(${MODULE_NAME} PRIVATE  glog::glog CUDA::cudart pybind11::embed)
install(TARGETS ${MODULE_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})