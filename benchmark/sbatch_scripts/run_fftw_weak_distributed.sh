#!/bin/bash

# Benchmark script for distributed memory weak scaling
# $1: Executable name
# $2: FFTW planning flag (estimate/measure)
# $3: Starting number of nodes
# $4: Stopping number of nodes
# $5: Base size
# $6: Number of threads per node
# $7: Number of runs
# $8: Partition

# Log Info
pwd; hostname; date
# Interleave the memory pages over the NUMA domains as a safety net: pages that one thread touches first would all sit in
# one domain (up to 10x slower at large sizes). Disable with NUMA_INTERLEAVE=0.
NUMA=""
if [[ "${NUMA_INTERLEAVE:-1}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
    NUMA="numactl --interleave=all"
fi
# FFTW wisdom: with measure/patient/exhaustive the first run of a configuration (example, size, threads, ranks) saves
# its wisdom in $WISDOM_DIR and the repeats load it, so every repeat uses the same plans and does not plan again. The
# directory must be on a shared file system for multi-node runs. Disable with WISDOM=0; estimate never uses wisdom.
if [[ "${WISDOM:-1}" == "1" && "$2" != "estimate" ]]; then
    export FFTW_WISDOM_DIR="${WISDOM_DIR:-$PWD/wisdom}"
fi
# Pin OpenMP threads to cores (inside the CPUs Slurm gives each task)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# Parameters
POW_START=$3
POW_STOP=$4
BASE_SIZE=$5
THREADS=$6
LOOP=$7
PARTITION=$8
# Get run command
COMMAND="srun --mpi=pmix -p $PARTITION -N $((2**POW_START)) -n $((2**POW_START)) -c $THREADS $NUMA"
EXECUTABLE=$1
ARGUMENTS="$BASE_SIZE $BASE_SIZE $2"
if [[ "$(basename "$EXECUTABLE")" == *"3d"* ]]; then
    ARGUMENTS="$BASE_SIZE $BASE_SIZE $BASE_SIZE $2"
fi
# Weak scaling loop from 2^pow_start to 2^pow_stop nodes
echo 'Submiting:' $COMMAND $THREADS $EXECUTABLE $ARGUMENTS
$COMMAND $EXECUTABLE $THREADS $ARGUMENTS 1
for (( j=1; j<$LOOP; j=j+1 ))
do
    echo 'Submiting:' $COMMAND $EXECUTABLE $THREADS $ARGUMENTS
    $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
done
for (( i=2**($POW_START+1); i<=2**$POW_STOP; i=i*2 ))
do
    COMMAND="srun --mpi=pmix -p $PARTITION -N $i -n $i -c $THREADS $NUMA"
    ARGUMENTS="$((BASE_SIZE*i)) $((BASE_SIZE*i)) $2"
    if [[ "$(basename "$EXECUTABLE")" == *"3d"* ]]; then
        ARGUMENTS="$((BASE_SIZE*i)) $((BASE_SIZE)) $BASE_SIZE $2"
    fi
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        echo 'Submiting:' $COMMAND $EXECUTABLE $THREADS $ARGUMENTS
        $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
    done
done
# Log Info
date
