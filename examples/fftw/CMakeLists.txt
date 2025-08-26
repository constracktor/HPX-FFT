cmake_minimum_required(VERSION 3.18)
project(fftw_example CXX)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

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

# MPI+OpenMP
find_package(OpenMP REQUIRED)
find_library(
    FFTW_OMP_MPI_LIB
    NAMES "fftw3_mpi"
    PATHS $ENV{FFTW_OMP_DIR}
)
find_library(
    FFTW_OMP_LIB
    NAMES "fftw3_omp"
    PATHS $ENV{FFTW_OMP_DIR}
)
add_executable(fftw_mpi_omp src/fftw_mpi_omp.cpp)
target_link_libraries(fftw_mpi_omp
                        PkgConfig::FFTW
                        ${FFTW_OMP_MPI_LIB}
                        ${FFTW_OMP_LIB}
                        OpenMP::OpenMP_CXX)
# OpenMP
add_executable(fftw_omp src/fftw_omp.cpp)
target_link_libraries(fftw_omp
                        PkgConfig::FFTW
                        ${FFTW_OMP_LIB}
                        OpenMP::OpenMP_CXX)
