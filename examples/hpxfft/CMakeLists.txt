cmake_minimum_required(VERSION 3.18)
project(hpxfft_examples CXX)

find_package(HPXFFT REQUIRED)

# 2D shared examples
add_executable(hpxfft_shared_loop_2d shared_loop_2d.cpp)
target_link_libraries(hpxfft_shared_loop_2d PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_sync_2d shared_sync_2d.cpp)
target_link_libraries(hpxfft_shared_sync_2d PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_opt_2d shared_opt_2d.cpp)
target_link_libraries(hpxfft_shared_opt_2d PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_naive_2d shared_naive_2d.cpp)
target_link_libraries(hpxfft_shared_naive_2d PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_agas_2d shared_agas_2d.cpp)
target_link_libraries(hpxfft_shared_agas_2d PRIVATE HPXFFT::hpxfft)

# 2D distributed examples
add_executable(hpxfft_distributed_loop_2d distributed_loop_2d.cpp)
target_link_libraries(hpxfft_distributed_loop_2d PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_distributed_agas_2d distributed_agas_2d.cpp)
target_link_libraries(hpxfft_distributed_agas_2d PRIVATE HPXFFT::hpxfft)

# 3D shared example
add_executable(hpxfft_shared_loop_3d shared_loop_3d.cpp)
target_link_libraries(hpxfft_shared_loop_3d PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_naive_3d shared_naive_3d.cpp)
target_link_libraries(hpxfft_shared_naive_3d PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_shared_sync_3d shared_sync_3d.cpp)
target_link_libraries(hpxfft_shared_sync_3d PRIVATE HPXFFT::hpxfft)

# 3D distributed examples
add_executable(hpxfft_distributed_loop_3d_pencil distributed_loop_3d_pencil.cpp)
target_link_libraries(hpxfft_distributed_loop_3d_pencil PRIVATE HPXFFT::hpxfft)

add_executable(hpxfft_distributed_loop_3d_slab distributed_loop_3d_slab.cpp)
target_link_libraries(hpxfft_distributed_loop_3d_slab PRIVATE HPXFFT::hpxfft)
