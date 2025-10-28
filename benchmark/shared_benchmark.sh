##!/usr/bin/bash
# Benchmark script for shared memory 
# $1: FFTW planning flag (estimate/measure)
################################################################################
# Config
################################################################################
TOP_DIR=$(pwd)
EXAMPLES_DIR=$TOP_DIR/examples

# HPX-FFT implementations
HPXFFT_EXECUTABLES=(
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_shared_naive"
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_shared_opt"
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_shared_sync"
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_shared_loop"
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_distributed_loop" 
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_shared_agas"
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_distributed_agas"
)
# FFTW implementations
FFTW_EXECUTABLES=(
    "$EXAMPLES_DIR/fftw/build/fftw_omp"
    "$EXAMPLES_DIR/fftw/build/fftw_threads"
    "$EXAMPLES_DIR/fftw/build/fftw_mpi_omp"
    "$EXAMPLES_DIR/fftw/build/fftw_mpi_threads"
)

# Get current hostname
HOSTNAME=$(hostname -s)
if [[ "$HOSTNAME" == "ipvsmisc" ]]; then
    echo "tbd."
elif [[ "$HOSTNAME" == "rostam" ]]; then
    echo "tbd."
elif [[ "$HOSTNAME" == "login1" ]]; then
    PARTITION=short
    THREAD_POW=5
elif [[ "$HOSTNAME" == "simcl1" ]]; then
    echo "tbd."
else
    echo "Hostname is $HOSTNAME — no action taken."
    exit 1
fi
THREADS=$((2**$THREAD_POW))

# Set FFTW planning
FFTW_PLAN=$1
# Check if FFTW_PLAN was provided
if [ -z "$FFTW_PLAN" ]; then
    echo "Error: FFTW_PLAN parameter not set."
    echo "Usage: shared_benchmark.sh estimate/measure"
    exit 1
fi
################################################################################
# Run benchmarks
################################################################################
RESULT_DIR=$TOP_DIR/shared_benchmark_on_$HOSTNAME
SCRIPT_DIR=$TOP_DIR/benchmark/sbatch_scripts

################################################################################
# Strong scaling
# Directories
mkdir -p $RESULT_DIR/strong_scaling/runtimes
mkdir -p $RESULT_DIR/strong_scaling/plans
cd $RESULT_DIR/strong_scaling

# Config
LOOP=2
BASE_SIZE=16384
START_THREAD_POW=3

# Loop over HPX-FFT executables
RUN_SCRIPT="$SCRIPT_DIR/run_hpxfft_strong_shared.sh"
for EXE in "${HPXFFT_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

  sbatch -p $PARTITION -N 1 -n 1 -c $THREADS \
    $RUN_SCRIPT \
    $EXE $FFTW_PLAN $START_THREAD_POW $THREAD_POW $BASE_SIZE $LOOP
done

# Loop over FFTW executables
RUN_SCRIPT="$SCRIPT_DIR/run_fftw_strong_shared.sh"
# Loop over executables
for EXE in "${FFTW_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

  sbatch -p $PARTITION -N 1 -n 1 -c $THREADS \
    $RUN_SCRIPT \
    $EXE $FFTW_PLAN $START_THREAD_POW $THREAD_POW $BASE_SIZE $LOOP
done

################################################################################
# Weak scaling
# Directories
mkdir -p $RESULT_DIR/weak_scaling/runtimes
mkdir -p $RESULT_DIR/weak_scaling/plans
cd $RESULT_DIR/weak_scaling

# Config
LOOP=2
BASE_SIZE=512
START_THREAD_POW=0

# Loop over HPX-FFT executables
RUN_SCRIPT="$SCRIPT_DIR/run_hpxfft_weak_shared.sh"
for EXE in "${HPXFFT_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

  sbatch -p $PARTITION -N 1 -n 1 -c $THREADS \
    $RUN_SCRIPT \
    $EXE $FFTW_PLAN $START_THREAD_POW $THREAD_POW $BASE_SIZE $LOOP
done

# Loop over FFTW executables
RUN_SCRIPT="$SCRIPT_DIR/run_fftw_weak_shared.sh"
# Loop over executables
for EXE in "${FFTW_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

  sbatch -p $PARTITION -N 1 -n 1 -c $THREADS \
    $RUN_SCRIPT \
    $EXE $FFTW_PLAN $START_THREAD_POW $THREAD_POW $BASE_SIZE $LOOP
done

################################################################################
# Problem size scaling
# Directories
mkdir -p $RESULT_DIR/size_scaling/runtimes
mkdir -p $RESULT_DIR/size_scaling/plans
cd $RESULT_DIR/size_scaling

# Config
LOOP=2
START_SIZE_POW=5
STOP_SIZE_POW=12
# Loop over HPX-FFT executables
RUN_SCRIPT="$SCRIPT_DIR/run_hpxfft_size_shared.sh"
for EXE in "${HPXFFT_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"
 
  sbatch -p $PARTITION -N 1 -n 1 -c $THREADS \
    $RUN_SCRIPT \
    $EXE $FFTW_PLAN $THREADS $START_SIZE_POW $STOP_SIZE_POW $LOOP
done

# Loop over FFTW executables
RUN_SCRIPT="$SCRIPT_DIR/run_fftw_size_shared.sh"
# Loop over executables
for EXE in "${FFTW_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"
 
  sbatch -p $PARTITION -N 1 -n 1 -c $THREADS \
    $RUN_SCRIPT \
    $EXE $FFTW_PLAN $THREADS $START_SIZE_POW $STOP_SIZE_POW $LOOP
done
