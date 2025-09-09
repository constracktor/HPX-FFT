cmake_minimum_required(VERSION 3.18)
project(hpxfft_examples CXX)

find_package(HPXFFT REQUIRED)

# Shared examples
add_executable(hpxfft_shared_loop shared_loop_2d.cpp)
target_link_libraries(hpxfft_shared_loop PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_sync shared_sync_2d.cpp)
target_link_libraries(hpxfft_shared_sync PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_opt shared_opt_2d.cpp)
target_link_libraries(hpxfft_shared_opt PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_naive shared_naive_2d.cpp)
target_link_libraries(hpxfft_shared_naive PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_agas shared_agas_2d.cpp)
target_link_libraries(hpxfft_shared_agas PRIVATE HPXFFT::hpxfft)

# Distributed examples
add_executable(hpxfft_distributed_loop distributed_loop_2d.cpp)
target_link_libraries(hpxfft_distributed_loop PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_distributed_agas distributed_agas_2d.cpp)
target_link_libraries(hpxfft_distributed_agas PRIVATE HPXFFT::hpxfft)
