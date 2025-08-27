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

add_library(hpxfft ../../core/src/shared/hpxfft_shared_loop.cpp
                   ../../core/src/shared/hpxfft_shared_sync.cpp
                   ../../core/src/shared/hpxfft_shared_opt.cpp
                   ../../core/src/shared/hpxfft_shared_naive.cpp
                   ../../core/src/shared/hpxfft_shared_agas.cpp)


target_include_directories(hpxfft PUBLIC ${CMAKE_SOURCE_DIR}/../../core/include/hpxfft)

target_link_libraries(hpxfft
                        HPX::hpx
                        PkgConfig::FFTW)

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
