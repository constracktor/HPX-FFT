#!/usr/bin/bash
# Benchmark script for message size scaling between two nodes
# $1: Collective (scatter/all_to_all)
################################################################################
# Config
################################################################################
TOP_DIR=$(pwd)
EXAMPLES_DIR=$TOP_DIR/examples

# HPX-FFT implementations
HPXFFT_EXECUTABLES=(
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_distributed_loop"
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_distributed_agas"
)
# FFTW implementations
FFTW_EXECUTABLES=(
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
    THREADS=48
elif [[ "$HOSTNAME" == "simcl1" ]]; then
    echo "tbd."
else
    echo "Hostname is $HOSTNAME — no action taken."
    exit 1
fi

# Set Collective
COLLECTIVE=$1
# Check if COLLECITVE was provided
if [ -z "$COLLECTIVE" ]; then
    echo "Error: COLLECTIVE parameter not set."
    echo "Usage: message_benchmark.sh scatter/all_to_all"
    exit 1
fi

################################################################################
# Run benchmarks
################################################################################
RESULT_DIR=$TOP_DIR/message_benchmark_on_$HOSTNAME
SCRIPT_DIR=$TOP_DIR/benchmark/sbatch_scripts

################################################################################
# Directories
mkdir -p $RESULT_DIR/runtimes
mkdir -p $RESULT_DIR/plans
cd $RESULT_DIR

# Config
LOOP=2
BASE_SIZE=128
STOP_POW=5

# Loop over HPX-FFT executables
RUN_SCRIPT="$SCRIPT_DIR/run_hpxfft_message_size.sh"
for EXE in "${HPXFFT_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

  sbatch -p $PARTITION -N 2 -n 2 -c $THREADS \
    $RUN_SCRIPT \
    $EXE $BASE_SIZE $STOP_POW $COLLECTIVE $THREADS $LOOP
done

# Loop over FFTW executables
RUN_SCRIPT="$SCRIPT_DIR/run_fftw_message_size.sh"
# Loop over executables
for EXE in "${FFTW_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

  sbatch -p $PARTITION -N 2 -n 2 -c $THREADS \
    $RUN_SCRIPT \
    $EXE $BASE_SIZE $STOP_POW $THREADS $LOOP
done
