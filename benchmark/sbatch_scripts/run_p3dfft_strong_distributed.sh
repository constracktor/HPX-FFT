#!/bin/bash
#SBATCH --error=p3dfft_error_%A.log    # Error Log

# Benchmark script for distributed memory strong scaling
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
# Pin OpenMP threads to cores (inside the CPUs Slurm gives each task)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# Sleeping instead of spinning idle OpenMP threads: with 48 spinning threads on 48 cores FFTW omp is up to 100x slower
export OMP_WAIT_POLICY=passive
# Parameters
NODES_START=$3
NODES_STOP=$4
BASE_SIZE=$5
THREADS=$6
LOOP=$7
PARTITION=$8
# Get run command
COMMAND="srun --mpi=pmix -p $PARTITION -N $((NODES_START**2)) -n $((NODES_START**2)) -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="$BASE_SIZE $BASE_SIZE $BASE_SIZE 2"
# Strong scaling loop from 2^pow_start to 2^pow_stop nodes
echo 'Submiting:' $COMMAND $EXECUTABLE $THREADS $ARGUMENTS
$COMMAND $EXECUTABLE $THREADS $ARGUMENTS 1
for (( j=1; j<$LOOP; j=j+1 ))
do
    echo 'Submiting:' $COMMAND $EXECUTABLE $THREADS $ARGUMENTS
    $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
done
for (( i=NODES_START+1; i<=NODES_STOP; i=i+1 ))
do
    SIZE=$((i**2))
    COMMAND="srun --mpi=pmix -p $PARTITION -N $SIZE -n $SIZE -c $THREADS"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        echo 'Submiting:' $COMMAND $EXECUTABLE $THREADS $ARGUMENTS
        $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
    done
done
# Log Info
date
