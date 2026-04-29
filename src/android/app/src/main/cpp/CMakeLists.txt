cmake_minimum_required(VERSION 3.22.1)
project(accompaniment_streaming_native)

add_library(
    accompaniment_streaming_native
    SHARED
    pitch_autocorr.cpp
    streaming_native.cpp
)

find_library(log-lib log)
target_link_libraries(accompaniment_streaming_native ${log-lib})

# Build native library with 16KB-compatible segment alignment for newer Android
# devices and emulator images using larger memory page sizes.
target_link_options(
    accompaniment_streaming_native
    PRIVATE
    "-Wl,-z,max-page-size=16384"
    "-Wl,-z,common-page-size=16384"
)
