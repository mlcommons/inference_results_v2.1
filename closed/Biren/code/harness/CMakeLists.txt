cmake_minimum_required(VERSION 3.10)
project(harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)


add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})
add_definitions(-DSUINFER_DIR=${SUINFER_DIR})

# Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})

find_library(SUDA NAMES libbrsuda.so)
# Find the static suinfer library
unset(SUINFER_LIB CACHE)
find_library(SUINFER_LIB NAMES libsuinfer.so)

# Set the path to the suinfer library
unset(SUINFER_INCLUDE_DIR CACHE)
set(SUINFER_INCLUDE_DIR ${SUINFER_DIR}/include)

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR br_perf_server/include)


# message(${SUINFER_DIR})
# message(${SUINFER_INCLUDE_DIR})
# message(${SUINFER_BUILD_DIR})
# message(${SUINFER_LIB})
# message(${LWIS_INCLUDE_DIR})
# message(${LOADGEN_LIB_DIR})
# message(${LOADGEN_LIB})


add_subdirectory(br_perf_server)

# Build the default harness which covers single_stream and offline scenarios on image benchmarks.
execute_process(COMMAND echo "Building default harness...")
add_executable(harness_default
        harness_default/main_default.cpp
        )

target_link_libraries(harness_default
        ${SUINFER_LIB}
        gflags
        glog
        br_perf_server
        ${LOADGEN_LIB}
        pthread
        )

target_include_directories(harness_default
        PUBLIC
        ${LWIS_INCLUDE_DIR}
        ${LOADGEN_INCLUDE_DIR}
        common
        )

execute_process(COMMAND echo "Building bert harness...")
add_executable(harness_bert
        harness_bert/main_bert.cpp
        )

target_link_libraries(harness_bert
        ${SUINFER_LIB}
        gflags
        glog
        br_perf_server
        ${LOADGEN_LIB}
	${SUDA}
        pthread
        )

target_include_directories(harness_bert
        PUBLIC
        ${LWIS_INCLUDE_DIR}
        ${LOADGEN_INCLUDE_DIR}
        common
        )
