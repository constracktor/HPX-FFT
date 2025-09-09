cmake_minimum_required(VERSION 3.18)
project(hpx_fft CXX)

find_package(HPX REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

set(SOURCE_FILES
                   ../../core/src/shared/loop.cpp
                   ../../core/src/shared/sync.cpp
                   ../../core/src/shared/opt.cpp
                   ../../core/src/shared/naive.cpp
                   ../../core/src/shared/agas.cpp
                   ../../core/src/distributed/loop.cpp
                   ../../core/src/distributed/agas.cpp)

add_library(hpxfft STATIC ${SOURCE_FILES})

# Add them as PRIVATE sources here so they show up in project files Can't use
# PUBLIC etc., see: https://stackoverflow.com/a/62465051
file(GLOB_RECURSE header_files CONFIGURE_DEPENDS ../../include/hpxfft/shared/*.hpp
                                                 ../../include/hpxfft/distributed/*.hpp)
target_sources(hpxfft PRIVATE ${header_files})

# Link HPX
target_link_libraries(hpxfft HPX::hpx)
# Link FFTW backend
target_link_libraries(hpxfft PkgConfig::FFTW)

# Include directories
target_include_directories(hpxfft PUBLIC ${CMAKE_SOURCE_DIR}/../../core/include)

target_compile_features(hpxfft PUBLIC cxx_std_20)

# Turn on PIC for shared libraries
set_property(TARGET hpxfft PROPERTY POSITION_INDEPENDENT_CODE ON)


#find_package(HPXFFT REQUIRED)

# Shared examples
add_executable(hpxfft_shared_loop shared_loop_2d.cpp)
target_link_libraries(hpxfft_shared_loop PRIVATE hpxfft)

add_executable(hpxfft_shared_sync shared_sync_2d.cpp)
target_link_libraries(hpxfft_shared_sync PRIVATE hpxfft)

add_executable(hpxfft_shared_opt shared_opt_2d.cpp)
target_link_libraries(hpxfft_shared_opt PRIVATE hpxfft)

add_executable(hpxfft_shared_naive shared_naive_2d.cpp)
target_link_libraries(hpxfft_shared_naive PRIVATE hpxfft)

add_executable(hpxfft_shared_agas shared_agas_2d.cpp)
target_link_libraries(hpxfft_shared_agas PRIVATE hpxfft)

# Distributed examples
add_executable(hpxfft_distributed_loop distributed_loop_2d.cpp)
target_link_libraries(hpxfft_distributed_loop PRIVATE hpxfft)

add_executable(hpxfft_distributed_agas distributed_agas_2d.cpp)
target_link_libraries(hpxfft_distributed_agas PRIVATE hpxfft)
