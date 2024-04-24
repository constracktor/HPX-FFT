#!/bin/bash
################################################################################
# Benchmark script for distributed memory 
# $1: partition (buran/medusa with mpi/lci/tcp/shmem)
# $2 communication scheme (scatter/all_to_all)
if [[ "$1" == "buran_mpi" ]] || [[ "$1" == "buran_lci" ]] || [[ "$1" == "buran_tcp" ]] || [[ "$1" == "buran_shmem" ]]
then
    # 16 nodes available
    module load llvm/17.0.1
    PARTITION=buran
    THREADS=48
    NODES_POW=4
elif [[ "$1" == "medusa_mpi" ]] || [[ "$1" == "medusa_lci" ]] || [[ "$1" == "medusa_tcp" ]] || [[ "$1" == "medusa_shmem" ]]
then
    # Only 14 nodes available
    module load llvm/17.0.1
    PARTITION=medusa
    THREADS=40
    NODES_POW=3
else
  echo 'Please specify partition and parcelport'
  exit 1
fi
LOOP=1
#50
FFTW_PLAN=measure
NODES=$((2**$NODES_POW))
COLLECTIVE=$2
BUILD_DIR=build_$1
# load modules
module load openmpi
cd benchmark
# HPX implementations
sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_hpx_dist.sh $BUILD_DIR/fft_hpx_loop $FFTW_PLAN $NODES_POW $COLLECTIVE $LOOP
#sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_hpx_dist.sh $BUILD_DIR/fft_hpx_task_agas $FFTW_PLAN $NODES_POW $COLLECTIVE $LOOP
# FFTW backends
#sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_fftw_dist.sh $BUILD_DIR/fftw_mpi_threads $FFTW_PLAN $NODES $THREADS $LOOP
#sbatch -p $PARTITION -N $NODES -n $NODES -c $THREADS run_fftw_dist.sh $BUILD_DIR/fftw_mpi_omp $FFTW_PLAN $NODES $THREADS $LOOP