cmake_minimum_required(VERSION 3.10)

set(module_preprocess suinfer_preprocess)
set(module_preprocess_simd suinfer_preprocess_simd)

# SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
# SET (CMAKE_CXX_FLAGS                "-Wall -pthread -march=native")
SET (CMAKE_CXX_FLAGS                "-pthread -march=native")
SET (CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG -s")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

project(suinfer_preprocess LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

option(OPEN_SIMD "open avx512" OFF)


set(PREPROCESS_SRC "")
list(APPEND PREPROCESS_SRC
                src/suinfer_preprocess.cpp
                src/utils.cpp
                )

set(PREPROCESS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${PREPROCESS_INCLUDE_DIR})

add_subdirectory(third-party)
include_directories(${FREEIMAGE_INCLUDE})
include_directories(${SUINFER_INCLUDE})

# add gflags v2.2.2
add_executable(${module_preprocess} ${PREPROCESS_SRC})
target_link_libraries(${module_preprocess}
    -lpthread
    ${FREEIMAGE_LIB}
    gflags
    )

