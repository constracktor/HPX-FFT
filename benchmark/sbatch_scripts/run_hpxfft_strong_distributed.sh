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
COMMAND="srun --mpi=pmix -p $PARTITION --nodelist=$PARTITION[00-0$((2**POW_START - 1))] -N $((2**POW_START)) -n $((2**POW_START)) -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="--nx=$BASE_SIZE --ny=$BASE_SIZE --plan=$2 --run=$6"
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
    COMMAND="srun --mpi=pmix -p $PARTITION --nodelist=$PARTITION[00-0$(($i - 1))] -N $i -n $i -c $THREADS"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
        HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
    done
done
# Log Info
date
