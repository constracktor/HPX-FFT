cmake_minimum_required(VERSION 3.18)
project(hpx_fft CXX)

find_package(HPX REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

###########################################################
# HPX code 
# add_executable(fft_hpx_loop core/src/shared/fft_hpx_loop.cpp)
# target_link_libraries(fft_hpx_loop
#                         HPX::hpx
#                         PkgConfig::FFTW)

# add_executable(fft_hpx_task_agas src/fft_hpx_task_agas.cpp)
# target_link_libraries(fft_hpx_task_agas
#                         HPX::hpx
#                         PkgConfig::FFTW)
#
# add_executable(fft_hpx_task_agas_shared src/fft_hpx_task_agas_shared.cpp)
# target_link_libraries(fft_hpx_task_agas_shared
#                         HPX::hpx
#                         PkgConfig::FFTW)
#
# add_executable(fft_hpx_task_sync_shared src/fft_hpx_task_sync_shared.cpp)
# target_link_libraries(fft_hpx_task_sync_shared
#                         HPX::hpx
#                         PkgConfig::FFTW)
#
# add_executable(fft_hpx_task_naive_shared src/fft_hpx_task_naive_shared.cpp)
# target_link_libraries(fft_hpx_task_naive_shared
#                         HPX::hpx
#                         PkgConfig::FFTW)
#
# add_executable(fft_hpx_task_shared src/fft_hpx_task_shared.cpp)
# target_link_libraries(fft_hpx_task_shared
#                         HPX::hpx
#                         PkgConfig::FFTW)
#

add_library(hpxfft ../../core/src/shared/hpxfft_loop_shared.cpp)

target_include_directories(hpxfft PUBLIC ${CMAKE_SOURCE_DIR}/../../core/include/shared/hpxfft_loop_shared.hpp)

target_link_libraries(hpxfft
                        HPX::hpx
                        PkgConfig::FFTW)

add_executable(hpxfft_loop_shared loop_shared_2d.cpp)
target_link_libraries(hpxfft_loop_shared PRIVATE hpxfft)
