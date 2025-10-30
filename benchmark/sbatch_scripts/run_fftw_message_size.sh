#!/bin/bash
#SBATCH --job-name=fftw_job          # Job name
#SBATCH --output=fftw_job.log        # Standard output and error log
#SBATCH --mail-type=NONE               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexander.strack@ipvs.uni-stuttgart.de       # Where to send mail	
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec
#SBATCH --exclusive                    # Exclusive ressource access

# Benchmark script for message size scaling between two nodes
# $1: Executable name
# $2: Base size
# $3: Stopping power of two
# $4: Number of threads per node
# $5: Number of runs

# Log Info
pwd; hostname; date
# Parameters
BASE_SIZE=$2
POW_STOP=$3
THREADS=$4
LOOP=$5
# Get run command
COMMAND="srun --mpi=pmix -N 2 -n 2 -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="$BASE_SIZE $BASE_SIZE $2"
# Message size scaling loop on 2 nodes
$COMMAND $EXECUTABLE $THREADS $ARGUMENTS 1
for (( j=1; j<$LOOP; j=j+1 ))
do
    $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
done
for (( i=2; i<=2**$POW_STOP; i=i*2 ))
do
    SIZE=$((i*BASE_SIZE))
    ARGUMENTS="$SIZE $SIZE $2"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
    done
done
# Log Info
date
