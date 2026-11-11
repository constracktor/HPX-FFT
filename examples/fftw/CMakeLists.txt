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
add_executable(fftw_mpi_threads_2d fftw_mpi_threads_2d.cpp)
target_link_libraries(fftw_mpi_threads_2d PkgConfig::FFTW ${FFTW_TH_MPI_LIB}
                      ${FFTW_TH_LIB} MPI::MPI_CXX)

add_executable(fftw_mpi_threads_3d fftw_mpi_threads_3d.cpp)
target_link_libraries(fftw_mpi_threads_3d PkgConfig::FFTW ${FFTW_TH_MPI_LIB}
                      ${FFTW_TH_LIB} MPI::MPI_CXX)

# Threads
add_executable(fftw_threads_2d fftw_threads_2d.cpp)
target_link_libraries(fftw_threads_2d PkgConfig::FFTW ${FFTW_TH_LIB})

add_executable(fftw_threads_3d fftw_threads_3d.cpp)
target_link_libraries(fftw_threads_3d PkgConfig::FFTW ${FFTW_TH_LIB})

# MPI+OpenMP
find_library(
  FFTW_OMP_MPI_LIB
  NAMES "fftw3_mpi"
  PATHS $ENV{FFTW_DIR})
find_library(
  FFTW_OMP_LIB
  NAMES "fftw3_omp"
  PATHS $ENV{FFTW_DIR})
add_executable(fftw_mpi_omp_2d fftw_mpi_omp_2d.cpp)
target_link_libraries(fftw_mpi_omp_2d PkgConfig::FFTW ${FFTW_OMP_MPI_LIB}
                      ${FFTW_OMP_LIB} OpenMP::OpenMP_CXX MPI::MPI_CXX)

add_executable(fftw_mpi_omp_3d fftw_mpi_omp_3d.cpp)
target_link_libraries(fftw_mpi_omp_3d PkgConfig::FFTW ${FFTW_OMP_MPI_LIB}
                      ${FFTW_OMP_LIB} OpenMP::OpenMP_CXX MPI::MPI_CXX)

# OpenMP
add_executable(fftw_omp_2d fftw_omp_2d.cpp)
target_link_libraries(fftw_omp_2d PkgConfig::FFTW ${FFTW_OMP_LIB}
                      OpenMP::OpenMP_CXX)

add_executable(fftw_omp_3d fftw_omp_3d.cpp)
target_link_libraries(fftw_omp_3d PkgConfig::FFTW ${FFTW_OMP_LIB}
                      OpenMP::OpenMP_CXX)

# P3DFFT
find_library(P3DFFT_LIB NAMES p3dfft.3 p3dfft)
find_path(P3DFFT_INCLUDE_DIR p3dfft.h)
# P3DFFT calls single-precision FFTW (fftwf_*), so link libfftw3f and
# libfftw3f_omp.
find_library(
  FFTW_SINGLE_OMP_LIB
  NAMES "fftw3f_omp"
  PATHS $ENV{FFTW_DIR})
find_library(
  FFTW_SINGLE_LIB
  NAMES "fftw3f"
  PATHS $ENV{FFTW_DIR})
add_executable(p3dfft_mpi_omp p3dfft_mpi_omp.cpp)
target_include_directories(p3dfft_mpi_omp PRIVATE ${P3DFFT_INCLUDE_DIR})
target_link_libraries(
  p3dfft_mpi_omp
  ${P3DFFT_LIB}
  PkgConfig::FFTW
  ${FFTW_OMP_LIB}
  ${FFTW_SINGLE_LIB}
  ${FFTW_SINGLE_OMP_LIB}
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
  gfortran
  m)
