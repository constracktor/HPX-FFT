#!/bin/bash
################################################################################
# Benchmark script for shared memory 
# $1: FFTW planning flag (estimate/measure)
# $2: partition (epyc/sven)
if [[ "$2" == "epyc" ]]
then
    PARTITION=epyc
    THREAD_POW=7
    BUILD_DIR=../build_epyc
elif [[ "$2" == "sven" ]]
then
    PARTITION="risc5 -w sven0"
    THREAD_POW=6
    BUILD_DIR=../build_sven
else
  echo 'Please specify partition'
  exit 1
fi
LOOP=50
FFTW_PLAN=$1
THREADS=$((2**$THREAD_POW))
cd sbatch_scripts
# HPX implementations
# shared only
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_hpx_shared.sh $BUILD_DIR/fft_hpx_loop_shared $FFTW_PLAN $THREAD_POW $LOOP
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_hpx_shared.sh $BUILD_DIR/fft_hpx_task_sync_shared $FFTW_PLAN $THREAD_POW $LOPP
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_hpx_shared.sh $BUILD_DIR/fft_hpx_task_shared $FFTW_PLAN $THREAD_POW $LOOP
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_hpx_shared.sh $BUILD_DIR/fft_hpx_task_naive_shared $FFTW_PLAN $THREAD_POW $LOOP
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_hpx_shared.sh $BUILD_DIR/fft_hpx_task_agas_shared $FFTW_PLAN $THREAD_POW $LOOP
# distributed possible
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_hpx_shared.sh $BUILD_DIR/fft_hpx_loop $FFTW_PLAN $THREAD_POW $LOOP
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_hpx_shared.sh $BUILD_DIR/fft_hpx_task_agas $FFTW_PLAN $THREAD_POW $LOOP
# FFTW backends
# shared only
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_fftw_shared.sh $BUILD_DIR/fftw_hpx $FFTW_PLAN $THREAD_POW $LOOP
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_fftw_shared.sh $BUILD_DIR/fftw_threads $FFTW_PLAN $THREAD_POW $LOOP
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_fftw_shared.sh $BUILD_DIR/fftw_omp $FFTW_PLAN $THREAD_POW $LOOP
# distributed possible
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_fftw_shared.sh $BUILD_DIR/fftw_mpi_threads $FFTW_PLAN $THREAD_POW $LOOP
sbatch -p $PARTITION -N 1 -n 1 -c $THREADS run_fftw_shared.sh $BUILD_DIR/fftw_mpi_omp $FFTW_PLAN $THREAD_POW $LOOP

