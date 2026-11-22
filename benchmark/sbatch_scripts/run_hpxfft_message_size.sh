#!/bin/bash
# Benchmark script for message size scaling between two nodes
# $1: Executable name
# $2: Base size
# $3: Stopping power of two
# $4: HPX collective (scatter/all_to_all)
# $5: Number of threads per node
# $6: Number of runs
# $7: Partition
# $8: HPX Parcelport

# Log Info
pwd; hostname; date
# Interleave the memory pages over the NUMA domains: the examples touch their input from one thread, so all pages
# would sit in one domain (up to 10x slower at large sizes). Disable with NUMA_INTERLEAVE=0.
NUMA=""
if [[ "${NUMA_INTERLEAVE:-1}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
    NUMA="numactl --interleave=all"
fi
# Parameters
BASE_SIZE=$2
POW_STOP=$3
THREADS=$5
LOOP=$6
PARTITION=$7
PARCELPORT=$8
# Get run command
OPTIONS=""
COMMAND="srun --mpi=pmix -p $PARTITION -N 2 -n 2 -c $THREADS $NUMA"
EXECUTABLE=$1
# the 2D agas component needs ny / 2 + 1 divisible by the number of localities: ny = size - 2 (nx = size)
NY_OFFSET=0
if [[ "$(basename "$EXECUTABLE")" == *"agas"* ]]; then
    NY_OFFSET=2
fi
ARGUMENTS="--nx=$BASE_SIZE --ny=$((BASE_SIZE - NY_OFFSET)) --plan="estimate" --run=$4"
if [[ "$(basename "$EXECUTABLE")" == *"3d"* ]]; then
    ARGUMENTS="--nx=$BASE_SIZE --ny=$BASE_SIZE --nz=$BASE_SIZE --plan="estimate" --run=$4"
fi
PARCELPORTS="--hpx:ini=hpx.parcel.mpi.enable=0 --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.lci.enable=0 --hpx:ini=hpx.parcel.$PARCELPORT.enable=1"
# Message size scaling loop on 2 nodes
echo 'Submiting warmup:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
$COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS --header=true
for (( j=0; j<$LOOP; j=j+1 ))
do
    echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
    $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
done
for (( i=2; i<=2**$POW_STOP; i=i*2 ))
do
    SIZE=$((i*BASE_SIZE))
    ARGUMENTS="--nx=$SIZE --ny=$((SIZE - NY_OFFSET)) --plan="estimate" --run=$4"
    if [[ "$(basename "$EXECUTABLE")" == *"3d"* ]]; then
        ARGUMENTS="--nx=$SIZE --ny=$SIZE --nz=$SIZE --plan="estimate" --run=$4"
    fi
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
        $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
    done
done
# Log Info
date
