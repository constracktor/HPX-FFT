cmake_minimum_required(VERSION 3.18)
project(hpx_fft CXX)

find_package(HPX REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

###########################################################
# HPX code 
add_executable(fft_hpx_loop src/fft_hpx_loop.cpp)
target_link_libraries(fft_hpx_loop
                        HPX::hpx
                        PkgConfig::FFTW)

# add_executable(fft_hpx_task_agas src/fft_hpx_task_agas.cpp)
# target_link_libraries(fft_hpx_task_agas
#                         HPX::hpx
#                         PkgConfig::FFTW)

# add_executable(fft_hpx_task_agas_shared src/fft_hpx_task_agas_shared.cpp)
# target_link_libraries(fft_hpx_task_agas_shared
#                         HPX::hpx
#                         PkgConfig::FFTW)

# add_executable(fft_hpx_task_sync_shared src/fft_hpx_task_sync_shared.cpp)
# target_link_libraries(fft_hpx_task_sync_shared
#                         HPX::hpx
#                         PkgConfig::FFTW)

# add_executable(fft_hpx_task_naive_shared src/fft_hpx_task_naive_shared.cpp)
# target_link_libraries(fft_hpx_task_naive_shared
#                         HPX::hpx
#                         PkgConfig::FFTW)

# add_executable(fft_hpx_task_shared src/fft_hpx_task_shared.cpp)
# target_link_libraries(fft_hpx_task_shared
#                         HPX::hpx
#                         PkgConfig::FFTW)

# add_executable(fft_hpx_loop_shared src/fft_hpx_loop_shared.cpp)
# target_link_libraries(fft_hpx_loop_shared
#                         HPX::hpx
#                         PkgConfig::FFTW)

###########################################################
# FFTW
# MPI+Threads
find_library(
    FFTW_TH_MPI_LIB
    NAMES "fftw3_mpi"
    PATHS $ENV{FFTW_TH_DIR}
)
find_library(
    FFTW_TH_LIB
    NAMES "fftw3_threads"
    PATHS $ENV{FFTW_TH_DIR}
)
add_executable(fftw_mpi_threads src/fftw_mpi_threads.cpp)
target_link_libraries(fftw_mpi_threads
                        PkgConfig::FFTW
                        ${FFTW_TH_MPI_LIB}
                        ${FFTW_TH_LIB})
# Threads
add_executable(fftw_threads src/fftw_threads.cpp)
target_link_libraries(fftw_threads
                        PkgConfig::FFTW
                        ${FFTW_TH_LIB})

# # MPI+OpenMP
# find_package(OpenMP REQUIRED)
# find_library(
#     FFTW_OMP_MPI_LIB
#     NAMES "fftw3_mpi"
#     PATHS $ENV{FFTW_OMP_DIR}
# )
# find_library(
#     FFTW_OMP_LIB
#     NAMES "fftw3_omp"
#     PATHS $ENV{FFTW_OMP_DIR}
# )
# add_executable(fftw_mpi_omp src/fftw_mpi_omp.cpp)
# target_link_libraries(fftw_mpi_omp
#                         PkgConfig::FFTW
#                         ${FFTW_OMP_MPI_LIB}
#                         ${FFTW_OMP_LIB}
#                         OpenMP::OpenMP_CXX)
# # OpenMP
# add_executable(fftw_omp src/fftw_omp.cpp)
# target_link_libraries(fftw_omp
#                         PkgConfig::FFTW
#                         ${FFTW_OMP_LIB}
#                         OpenMP::OpenMP_CXX)

# # HPX
# find_library(
#     FFTW_HPX_LIB
#     NAMES "fftw3_hpx"
#     PATHS $ENV{FFTW_HPX_DIR}
# )
# add_executable(fftw_hpx src/fftw_hpx.cpp)
# target_link_libraries(fftw_hpx
#                         PkgConfig::FFTW
#                         ${FFTW_HPX_LIB}
#                         HPX::hpx)