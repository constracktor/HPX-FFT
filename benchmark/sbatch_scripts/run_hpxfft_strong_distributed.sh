#!/bin/bash

# Benchmark script for distributed memory strong scaling
# $1: Executable name
# $2: FFTW planning flag (estimate/measure)
# $3: Starting number of nodes
# $4: Stopping number of nodes
# $5: Base size
# $6: HPX collective (scatter/all_to_all)
# $7: Number of threads per node
# $8: Number of runs
# $9: Partition
# $10: HPX Parcelport

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
# Parameters
POW_START=$3
POW_STOP=$4
BASE_SIZE=$5
THREADS=$7
LOOP=$8
PARTITION=$9
PARCELPORT=${10}
# Get run command
OPTIONS=""
COMMAND="srun --mpi=pmix -p $PARTITION -N $((2**POW_START)) -n $((2**POW_START)) -c $THREADS $NUMA"
EXECUTABLE=$1
# the 2D agas component needs ny / 2 + 1 divisible by the number of localities: ny = size - 2 (nx = size)
NY_OFFSET=0
if [[ "$(basename "$EXECUTABLE")" == *"agas"* ]]; then
    NY_OFFSET=2
fi
ARGUMENTS="--nx=$BASE_SIZE --ny=$((BASE_SIZE - NY_OFFSET)) --plan=$2 --run=$6"
if [[ "$(basename "$EXECUTABLE")" == *"3d"* ]]; then
    ARGUMENTS="--nx=$BASE_SIZE --ny=$BASE_SIZE --nz=$BASE_SIZE --plan=$2 --run=$6"
fi
PARCELPORTS="--hpx:ini=hpx.parcel.mpi.enable=0 --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.lci.enable=0 --hpx:ini=hpx.parcel.$PARCELPORT.enable=1"
# Strong scaling loop from 2^pow_start to 2^pow_stop nodes
echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS --header=true
for (( j=1; j<$LOOP; j=j+1 ))
do
    echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
    HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
done
for (( i=2**($POW_START+1); i<=2**$POW_STOP; i=i*2 ))
do
    COMMAND="srun --mpi=pmix -p $PARTITION -N $i -n $i -c $THREADS $NUMA"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
        HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
    done
done
# Log Info
date
