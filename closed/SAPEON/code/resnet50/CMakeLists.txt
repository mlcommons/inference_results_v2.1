cmake_minimum_required(VERSION 3.5)
project(SapeonMlperf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall")
message(STATUS "CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}") 
message(STATUS "Using C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Using C++ compiler flags: ${CMAKE_CXX_FLAGS}")

set(TARGET_NAME sapeon_inferencer)
set(LOADGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/MlperfInference/loadgen)
add_subdirectory(${LOADGEN_DIR} mlperf_loadgen)

# set(SAPEONRT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SapeonRT)
# add_subdirectory(${SAPEONRT_DIR})

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SUBMISSION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../submission)

add_executable(${TARGET_NAME}
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/mlperf_inferencer_x220.cpp
    ${SRC_DIR}/mlperf_inferencer_e1.cpp
)

# set(RUNTIME_BUILD_DIR ${SAPEONRT_DIR}/)
set(RUNTIME_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/libsapeonruntime.so.0)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include ${LOADGEN_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE mlperf_loadgen ${RUNTIME_LIB} ${CMAKE_DL_LIBS} glog pthread pthread protobuf stdc++fs)

install(
    TARGETS ${TARGET_NAME} 
    RUNTIME DESTINATION ${SUBMISSION_DIR}/closed/Sapeon/code/resnet50/build
)
