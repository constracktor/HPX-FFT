#!/usr/bin/bash
# Benchmark script for message size scaling between two nodes
# $1: Collective (scatter/all_to_all)
# $2: HPX Parcelport (tcp/mpi/lci)
################################################################################
# Config
################################################################################
TOP_DIR=$(pwd)
EXAMPLES_DIR=$TOP_DIR/examples

# HPX-FFT implementations
HPXFFT_EXECUTABLES=(
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_distributed_loop_3d_slab"
)

# Get current hostname
HOSTNAME=$(hostname -s)
if [[ "$HOSTNAME" == "rostam1" ]]; then
    PARTITION=buran
    THREADS=48
    module load gcc/14.2.0
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
# Set HPX Parcelport
PARCELPORT=$2
# Check if COLLECITVE was provided
if [ -z "$PARCELPORT" ]; then
    echo "Error: Parcelport parameter not set."
    echo "Usage: message_benchmark.sh scatter/all_to_all tcp/mpi/lci"
    exit 1
fi
################################################################################
# Run benchmarks
################################################################################
RESULT_DIR=$TOP_DIR/message_benchmark_on_$PARTITION
SCRIPT_DIR=$TOP_DIR/benchmark/sbatch_scripts

################################################################################
# Directories
mkdir -p $RESULT_DIR/runtimes
mkdir -p $RESULT_DIR/plans
cd $RESULT_DIR

# Config
LOOP=2
BASE_SIZE=32
STOP_POW=4

# Loop over HPX-FFT executables
RUN_SCRIPT="$SCRIPT_DIR/run_hpxfft_message_size.sh"
for EXE in "${HPXFFT_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

    $RUN_SCRIPT \
    $EXE $BASE_SIZE $STOP_POW $COLLECTIVE $THREADS $LOOP $PARTITION $PARCELPORT
done
