#!/bin/bash
################################################################################
# Benchmark script for distributed memory 
# $1: partition (buran/medusa with mpi/lci/tcp/shmem)
# $2 communication scheme (scatter/all_to_all)
if [[ "$1" == "buran_mpi" ]]
then
    # 16 nodes available
    module load llvm/17.0.1
    module load openmpi
    PARTITION=buran
    THREADS=48
    NODES_POW=4
elif [[ "$1" == "buran_lci" ]]
then
    # 16 nodes available
    module load llvm/17.0.1
    PARTITION=buran
    THREADS=48
    NODES_POW=4
    export LD_LIBRARY_PATH=/home/alex/hpxsc_installations/hpx_1.9_lci_clang_17.0.1/install/lib64:$LD_LIBRARY_PATH
elif [[ "$1" == "buran_tcp" ]] || [[ "$1" == "buran_shmem" ]]
then
    # 16 nodes available
    module load llvm/17.0.1
    PARTITION=buran
    THREADS=48
    NODES_POW=4
elif [[ "$1" == "medusa_mpi" ]]
then
    # 14 nodes available
    module load llvm/17.0.1
    module load openmpi
    PARTITION=medusa
    THREADS=40
    NODES_POW=3
elif [[ "$1" == "medusa_lci" ]]
then
    # 14 nodes available
    module load llvm/17.0.1
    PARTITION=medusa
    THREADS=40
    NODES_POW=3
    export LD_LIBRARY_PATH=/home/alex/hpxsc_installations/hpx_1.9_lci_clang_17.0.1_medusa/install/lib64:$LD_LIBRARY_PATH
elif [[ "$1" == "medusa_tcp" ]] || [[ "$1" == "medusa_shmem" ]]
then
    # 14 nodes available
    module load llvm/17.0.1
    PARTITION=medusa
    THREADS=40
    NODES_POW=3
else
  echo 'Please specify partition and parcelport'
  exit 1
fi
LOOP=50
FFTW_PLAN=measure
NODES=$((2**$NODES_POW))
COLLECTIVE=$2
BUILD_DIR=../build_$1
cd sbatch_scripts
# HPX implementations
sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_hpx_dist.sh $BUILD_DIR/fft_hpx_loop $FFTW_PLAN $NODES_POW $COLLECTIVE $LOOP
#sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_hpx_dist.sh $BUILD_DIR/fft_hpx_task_agas $FFTW_PLAN $NODES_POW $COLLECTIVE $LOOP
# FFTW backends
#sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_fftw_dist.sh $BUILD_DIR/fftw_mpi_threads $FFTW_PLAN $NODES $THREADS $LOOP
#sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_fftw_dist.sh $BUILD_DIR/fftw_mpi_omp $FFTW_PLAN $NODES $THREADS $LOOP
