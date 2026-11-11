#!/bin/bash
#SBATCH --error=hpxfft_error_%A.log    # Error Log

# Benchmark script for distributed memory strong scaling for pencil decomposition
# $1: Executable name
# $2: FFTW planning flag (estimate/measure)
# $3: Starting square number of nodes
# $4: Stopping square number of nodes
# $5: Base size
# $6: HPX collective (scatter/all_to_all)
# $7: Number of threads per node
# $8: Number of runs
# $9: Partition
# $10: HPX Parcelport

# Log Info
pwd; hostname; date
# Parameters
NODES_START=$3
NODES_STOP=$4
BASE_SIZE=$5
THREADS=$7
LOOP=$8
PARTITION=$9
PARCELPORT=${10}
# Get run command
OPTIONS=""
COMMAND="srun --mpi=pmix -p $PARTITION --nodelist=$PARTITION[$(printf '%02d' $((16 - 2**POW_START)))-15] -N $((NODES_START**2)) -n $((NODES_START**2)) -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="--nx=$BASE_SIZE --ny=$BASE_SIZE --nz=$((BASE_SIZE-2)) --plan=$2 --run=$6"
PARCELPORTS="--hpx:ini=hpx.parcel.mpi.enable=0 --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.lci.enable=0 --hpx:ini=hpx.parcel.$PARCELPORT.enable=1"
# Strong scaling loop from 2^pow_start to 2^pow_stop nodes
echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS --header=true
for (( j=1; j<$LOOP; j=j+1 ))
do
    echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
    HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
done
for (( i=NODES_START+1; i<=NODES_STOP; i=i+1 ))
do
    SIZE=$((i**2))
    COMMAND="srun --mpi=pmix -p $PARTITION --nodelist=$PARTITION[$(printf '%02d' $((16 - SIZE)))-15] -N $SIZE -n $SIZE -c $THREADS"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
        HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
    done
done
# Log Info
date
