#!/usr/bin/bash
# Benchmark script for distributed memory 
# $1: FFTW planning flag (estimate/measure)
# $2: Collective (scatter/all_to_all)
# $3: HPX Parcelport (tcp/mpi/lci)
################################################################################
# Config
################################################################################
TOP_DIR=$(pwd)
EXAMPLES_DIR=$TOP_DIR/examples

# HPX-FFT implementations
HPXFFT_EXECUTABLES=(
    "$EXAMPLES_DIR/hpxfft/build/hpxfft_distributed_loop_3d_slab"
)
HPXFFT_PENCIL_EXE="$EXAMPLES_DIR/hpxfft/build/hpxfft_distributed_loop_3d_pencil"
# FFTW implementations
FFTW_EXECUTABLES=(
    "$EXAMPLES_DIR/fftw/build/fftw_mpi_omp_3d"
    "$EXAMPLES_DIR/fftw/build/fftw_mpi_threads_3d"
)
FFTW_PENCIL_EXE="$EXAMPLES_DIR/fftw/build/p3dfft_mpi_omp"

# Get current hostname
HOSTNAME=$(hostname -s)
if [[ "$HOSTNAME" == "rostam1" ]]; then
    PARTITION=buran
    START_NODES_POW=0
    NODES_POW=4
    THREADS=48
    START_NODES=1
    END_NODES=4
    module load gcc/14.2.0
else
    echo "Hostname is $HOSTNAME — no action taken."
    exit 1
fi
NODES=$((2**NODES_POW))

# Set FFTW planning
FFTW_PLAN=$1
# Check if FFTW_PLAN was provided
if [ -z "$FFTW_PLAN" ]; then
    echo "Error: FFTW_PLAN parameter not set."
    echo "Usage: distributed_benchmark.sh estimate/measure scatter/all_to_all tcp/mpi/lci"
    exit 1
fi
# Set Collective
COLLECTIVE=$2
# Check if COLLECITVE was provided
if [ -z "$COLLECTIVE" ]; then
    echo "Error: COLLECTIVE parameter not set."
    echo "Usage: distributed_benchmark.sh estimate/measure scatter/all_to_all tcp/mpi/lci"
    exit 1
fi
# Set HPX Parcelport
PARCELPORT=$3
# Check if COLLECITVE was provided
if [ -z "$PARCELPORT" ]; then
    echo "Error: Parcelport parameter not set."
    echo "Usage: distributed_benchmark.sh estimate/measure scatter/all_to_all tcp/mpi/lci"
    exit 1
fi

################################################################################
# Run benchmarks
################################################################################
RESULT_DIR=$TOP_DIR/distributed_benchmark_on_$PARTITION
SCRIPT_DIR=$TOP_DIR/benchmark/sbatch_scripts

################################################################################
# Strong scaling
# Directories
mkdir -p $RESULT_DIR/strong_scaling/runtimes
mkdir -p $RESULT_DIR/strong_scaling/plans
cd $RESULT_DIR/strong_scaling

# Config
LOOP=10
BASE_SIZE=512
BASE_SIZE_PENCIL=576

# Loop over HPX-FFT executables
RUN_SCRIPT="$SCRIPT_DIR/run_hpxfft_strong_distributed.sh"
for EXE in "${HPXFFT_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"
  echo "Submitting $EXE $FFTW_PLAN $START_NODES_POW $NODES_POW $BASE_SIZE $COLLECTIVE $THREADS $LOOP $PARTITION $PARCELPORT"

  $RUN_SCRIPT \
    $EXE $FFTW_PLAN $START_NODES_POW $NODES_POW $BASE_SIZE $COLLECTIVE $THREADS $LOOP $PARTITION $PARCELPORT
done
# Loop over FFTW executables
RUN_SCRIPT="$SCRIPT_DIR/run_fftw_strong_distributed.sh"
# Loop over executables
for EXE in "${FFTW_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

  $RUN_SCRIPT \
    $EXE $FFTW_PLAN $START_NODES_POW $NODES_POW $BASE_SIZE $THREADS $LOOP $PARTITION
done

# Strong Scaling Pencil Decomposition

# Loop over HPX-FFT executables
RUN_SCRIPT="$SCRIPT_DIR/run_hpxfft_pencil_strong_distributed.sh"
echo "Submitting job for executable: $HPXFFT_PENCIL_EXE"
$RUN_SCRIPT \
  $HPXFFT_PENCIL_EXE $FFTW_PLAN $START_NODES $END_NODES $BASE_SIZE_PENCIL $COLLECTIVE $THREADS $LOOP $PARTITION $PARCELPORT


# Loop over FFTW executables
RUN_SCRIPT="$SCRIPT_DIR/run_p3dfft_strong_distributed.sh"
# Loop over executables
echo "Submitting job for executable: $FFTW_PENCIL_EXE"

$RUN_SCRIPT \
  $FFTW_PENCIL_EXE $FFTW_PLAN $START_NODES $END_NODES $BASE_SIZE_PENCIL $THREADS $LOOP $PARTITION

################################################################################
# Weak scaling
# Directories
mkdir -p $RESULT_DIR/weak_scaling/runtimes
mkdir -p $RESULT_DIR/weak_scaling/plans
cd $RESULT_DIR/weak_scaling

# Config
LOOP=10
BASE_SIZE=512
START_THREAD_POW=0

# Loop over HPX-FFT executables
RUN_SCRIPT="$SCRIPT_DIR/run_hpxfft_weak_distributed.sh"
for EXE in "${HPXFFT_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

  $RUN_SCRIPT \
    $EXE $FFTW_PLAN $START_NODES_POW $NODES_POW $BASE_SIZE $COLLECTIVE $THREADS $LOOP $PARTITION $PARCELPORT
done

# Loop over FFTW executables
RUN_SCRIPT="$SCRIPT_DIR/run_fftw_weak_distributed.sh"
# Loop over executables
for EXE in "${FFTW_EXECUTABLES[@]}"; do
  echo "Submitting job for executable: $EXE"

  $RUN_SCRIPT \
    $EXE $FFTW_PLAN $START_NODES_POW $NODES_POW $BASE_SIZE $THREADS $LOOP $PARTITION
done
