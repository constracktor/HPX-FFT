cmake_minimum_required(VERSION 3.18)
project(fftw_examples CXX)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

# MPI+POSIX Threads
find_library(
  FFTW_TH_MPI_LIB
  NAMES "fftw3_mpi"
  PATHS $ENV{FFTW_TH_DIR})
find_library(
  FFTW_TH_LIB
  NAMES "fftw3_threads"
  PATHS $ENV{FFTW_TH_DIR})
add_executable(fftw_mpi_threads fftw_mpi_threads_2d.cpp)
target_link_libraries(fftw_mpi_threads PkgConfig::FFTW ${FFTW_TH_MPI_LIB}
                      ${FFTW_TH_LIB} MPI::MPI_CXX)
# Threads
add_executable(fftw_threads fftw_threads_2d.cpp)
target_link_libraries(fftw_threads PkgConfig::FFTW ${FFTW_TH_LIB})

# MPI+OpenMP
find_library(
  FFTW_OMP_MPI_LIB
  NAMES "fftw3_mpi"
  PATHS $ENV{FFTW_DIR})
find_library(
  FFTW_OMP_LIB
  NAMES "fftw3_omp"
  PATHS $ENV{FFTW_DIR})
add_executable(fftw_mpi_omp fftw_mpi_omp_2d.cpp)
target_link_libraries(fftw_mpi_omp PkgConfig::FFTW ${FFTW_OMP_MPI_LIB}
                      ${FFTW_OMP_LIB} OpenMP::OpenMP_CXX MPI::MPI_CXX)

# OpenMP
add_executable(fftw_omp fftw_omp_2d.cpp)
target_link_libraries(fftw_omp PkgConfig::FFTW ${FFTW_OMP_LIB}
                      OpenMP::OpenMP_CXX)
