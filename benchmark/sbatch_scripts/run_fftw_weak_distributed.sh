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
# Parameters
POW_START=$3
POW_STOP=$4
BASE_SIZE=$5
THREADS=$6
LOOP=$7
PARTITION=$8
# Get run command
COMMAND="srun --mpi=pmix -p $PARTITION --nodelist=$PARTITION[00-0$((2**POW_START - 1))] -N $((2**POW_START)) -n $((2**POW_START)) -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="$BASE_SIZE $BASE_SIZE $2"
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
    COMMAND="srun --mpi=pmix -p $PARTITION --nodelist=$PARTITION[00-0$(($i - 1))] -N $i -n $i -c $THREADS"
    ARGUMENTS="$((BASE_SIZE*i)) $((BASE_SIZE*i)) $2"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        echo 'Submiting:' $COMMAND $EXECUTABLE $THREADS $ARGUMENTS
        $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
    done
done
# Log Info
date
