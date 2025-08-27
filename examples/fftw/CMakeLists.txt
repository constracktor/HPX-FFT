cmake_minimum_required(VERSION 3.18)
project(fftw_example CXX)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

# Check for FFTW3 (serial)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)

# Check for FFTW3 MPI extension
pkg_check_modules(FFTW3MPI REQUIRED IMPORTED_TARGET fftw3_mpi)
# # MPI+Threads
# find_library(
#     FFTW_TH_MPI_LIB
#     NAMES "fftw3_mpi"
#     PATHS $ENV{FFTW_TH_DIR}
# )
# find_library(
#     FFTW_TH_LIB
#     NAMES "fftw3_threads"
#     PATHS $ENV{FFTW_TH_DIR}
# )
# add_executable(fftw_mpi_threads src/fftw_mpi_threads.cpp)
# target_link_libraries(fftw_mpi_threads
#                         PkgConfig::FFTW
#                         ${FFTW_TH_MPI_LIB}
#                         ${FFTW_TH_LIB})
# # Threads
# add_executable(fftw_threads src/fftw_threads.cpp)
# target_link_libraries(fftw_threads
#                         PkgConfig::FFTW
#                         ${FFTW_TH_LIB})

# MPI+OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
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
add_executable(fftw_mpi_omp_2d fftw_mpi_omp_2d.cpp)
target_link_libraries(fftw_mpi_omp_2d
                        PkgConfig::FFTW
                        OpenMP::OpenMP_CXX
                        MPI::MPI_CXX)
# OpenMP
add_executable(fftw_omp_2d fftw_omp_2d.cpp)
target_link_libraries(fftw_omp_2d
                        PkgConfig::FFTW
                        OpenMP::OpenMP_CXX)
