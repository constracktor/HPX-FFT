#!/bin/bash
################################################################################
# Diagnostics
################################################################################
set +x

################################################################################
# Preprocessing
################################################################################
# working directory
export HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/.."
# export paths
if [[ "$1" == "epyc" ]]
then
    # epyc
    module load cmake
    export BUILD_DIR=build_$1
    export FFTW_DIR='.'
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_apex_epyc2_v.1.9.1/build/hpx/build/lib"
    # MPI compiler
    export CXX="${HOME}/hpxsc_installations/hpx_apex_epyc2_v.1.9.1/build/openmpi/bin/mpic++"
    #spack load cmake
    #module load gcc/13.2.0
    #export HPX_DIR="/home/strackar/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-13.2.0/hpx-1.9.1-hpgdg5dzwzolmnngayqs462ela3iboy6/lib"
elif [[ "$1" == "buran_mpi" ]]
then
    # buran with HPX using MPI parcelport
    export BUILD_DIR=build_$1
    export FFTW_DIR=buran
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_1.9_mpi_clang_17.0.1/install/lib64"
    module load llvm/17.0.1
    module load openmpi
    # Required flags to also compile the distributed FFTW implementations - BREAKS fftw_hpx backend
    export LDFLAGS="-pthread -L/opt/apps/openmpi/4.1.5/lib -L/opt/apps/hwloc/2.9.1/lib -Wl,-rpath -Wl,/opt/apps/openmpi/4.1.5/lib -Wl,-rpath -Wl,/opt/apps/hwloc/2.9.1/lib -Wl,--enable-new-dtags -lmpi"



    # module load gcc/11.2.1
    # # HPX lib directory
    # export HPX_DIR="${HOME}/hpxsc_installations/hpx_1.9.1_mpi_gcc_11.2.1/build/hpx/build/lib"
    # # MPI compiler
    # export CXX="${HOME}/hpxsc_installations/hpx_1.9.1_mpi_gcc_11.2.1/build/openmpi/bin/mpic++"
    # module load openmpi
elif [[ "$1" == "buran_lci" ]]
then
    # buran with HPX using LCI parcelport
    export BUILD_DIR=build_$1
    export FFTW_DIR=buran
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_1.9_lci_clang_17.0.1/install/lib64"
    module load llvm/17.0.1
elif [[ "$1" == "buran_tcp" ]]
then
    # buran with HPX using LCI parcelport
    export BUILD_DIR=build_$1
    export FFTW_DIR=buran
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_1.9_tcp_clang_17.0.1/install/lib64"
    module load llvm/17.0.1
elif [[ "$1" == "buran_shmem" ]]
then
    # buran with HPX using openshmem parcelport
    export BUILD_DIR=build_$1
    export FFTW_DIR=buran_clang
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_dev_shmem_clang_17.0.1/install/lib64"
    module load llvm/17.0.1
    module load openmpi
    export C_COMPILER=/opt/apps/llvm/17.0.1/bin/clang
# elif [[ "$1" == "buran_gasnet" ]]
# then
#     # buran with gasnet
#     export BUILD_DIR=build_$1
#     export FFTW_DIR=buran
#     module load llvm/17.0.1
#     # HPX lib directory
#     export HPX_DIR="${HOME}/hpxsc_installations/hpx_gasnet/install/lib64"
#     export PKG_CONFIG_PATH="${HOME}/hpxsc_installations/hpx_gasnet/gasnet_2023.9.0/install/lib/pkgconfig":$PKG_CONFIG_PATH
#     # MPI compiler
#     #export CXX="${HOME}/hpxsc_installations/hpx_1.9.1_mpi_gcc_11.2.1/build/openmpi/bin/mpic++" 
#     module load openmpi
elif [[ "$1" == "medusa_mpi" ]]
then
    # buran with HPX using MPI parcelport
    export BUILD_DIR=build_$1
    export FFTW_DIR=buran
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_1.9_mpi_clang_17.0.1_medusa/install/lib64"
    module load llvm/17.0.1
    module load openmpi
    # Required flags to also compile the distributed FFTW implementations - BREAKS fftw_hpx backend
    export LDFLAGS="-pthread -L/opt/apps/openmpi/4.1.5/lib -L/opt/apps/hwloc/2.9.1/lib -Wl,-rpath -Wl,/opt/apps/openmpi/4.1.5/lib -Wl,-rpath -Wl,/opt/apps/hwloc/2.9.1/lib -Wl,--enable-new-dtags -lmpi"
elif [[ "$1" == "medusa_lci" ]]
then
    # buran with HPX using LCI parcelport
    export BUILD_DIR=build_$1
    export FFTW_DIR=buran
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_1.9_lci_clang_17.0.1_medusa/install/lib64"
    module load llvm/17.0.1
elif [[ "$1" == "medusa_tcp" ]]
then
    # buran with HPX using LCI parcelport
    export BUILD_DIR=build_$1
    export FFTW_DIR=buran
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_1.9_tcp_clang_17.0.1_medusa/install/lib64"
    module load llvm/17.0.1
elif [[ "$1" == "medusa_shmem" ]]
then
    # medusa with HPX using openshmem parcelport
    export BUILD_DIR=build_$1
    export FFTW_DIR=buran_clang
    module load llvm/17.0.1
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_openshmem_medusa/install/lib64"
    module load openmpi
    export C_COMPILER=/opt/apps/llvm/17.0.1/bin/clang
elif [[ "$1" == "sven" ]]
then
    # sven
    export BUILD_DIR=build_$1
    export FFTW_DIR=sven
    # HPX lib directory
    export HPX_DIR="${HOME}/hpxsc_installations/hpx_1.9.1_gcc_13.2.1_sven/build/hpx/lib64"
    # MPI compiler
    module load mpi/openmpi-riscv64
    export CXX=mpic++
else
  echo 'Please specify system and parcelport'
  exit 1
fi
export CMAKE_COMMAND=cmake
# FFTW paths
export FFTW_TH_DIR="$HOME/fft_installations/$FFTW_DIR/fftw_threads_mpi/install/lib"
export FFTW_OMP_DIR="$HOME/fft_installations/$FFTW_DIR/fftw_omp_mpi/install/lib"
export FFTW_HPX_DIR="$HOME/fft_installations/$FFTW_DIR/fftw_hpx/install/lib"
# for sven with measure planning: export FFTW_HPX_DIR="$HOME/fft_installations/$FFTW_DIR/fftw_hpx_riscv/install/lib"
export PKG_CONFIG_PATH="$FFTW_TH_DIR/pkgconfig":$PKG_CONFIG_PATH

################################################################################
# Compilation
################################################################################
rm -rf $BUILD_DIR && mkdir $BUILD_DIR && cd $BUILD_DIR
LDFLAGS=$LDFLAGS CC=$C_COMPILER $CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DHPX_DIR="${HPX_DIR}/cmake/HPX"
make -j $(grep -c ^processor /proc/cpuinfo)
