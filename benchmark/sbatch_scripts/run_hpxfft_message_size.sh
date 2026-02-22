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
# Parameters
BASE_SIZE=$2
POW_STOP=$3
THREADS=$5
LOOP=$6
PARTITION=$7
PARCELPORT=$8
# Get run command
OPTIONS=""
COMMAND="srun --mpi=pmix -p $PARTITION --nodelist=$PARTITION[00-01] -N 2 -n 2 -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="--nx=$BASE_SIZE --ny=$BASE_SIZE --plan="estimate" --run=$4"
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
    ARGUMENTS="--nx=$SIZE --ny=$SIZE --plan="estimate" --run=$4"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
        $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
    done
done
# Log Info
date
