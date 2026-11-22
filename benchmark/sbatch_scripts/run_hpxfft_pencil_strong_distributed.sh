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
# Interleave the memory pages over the NUMA domains: the examples touch their input from one thread, so all pages
# would sit in one domain (up to 10x slower at large sizes). Disable with NUMA_INTERLEAVE=0.
NUMA=""
if [[ "${NUMA_INTERLEAVE:-1}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
    NUMA="numactl --interleave=all"
fi
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
COMMAND="srun --mpi=pmix -p $PARTITION -N $((NODES_START**2)) -n $((NODES_START**2)) -c $THREADS $NUMA"
EXECUTABLE=$1
ARGUMENTS="--nx=$BASE_SIZE --ny=$BASE_SIZE --nz=$BASE_SIZE --plan=$2 --run=$6"
PARCELPORTS="--hpx:ini=hpx.parcel.mpi.enable=0 --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.lci.enable=0 --hpx:ini=hpx.parcel.$PARCELPORT.enable=1"
# Strong scaling loop from NODES_START^2 to NODES_STOP^2 nodes
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
    COMMAND="srun --mpi=pmix -p $PARTITION -N $SIZE -n $SIZE -c $THREADS $NUMA"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        echo 'Submiting:' $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
        HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS $PARCELPORTS
    done
done
# Log Info
date
